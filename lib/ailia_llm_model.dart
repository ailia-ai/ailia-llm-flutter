import 'dart:ffi';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:io';
import 'package:ffi/ffi.dart';

import 'ailia_llm.dart' as ailia_llm_dart;

const String BACKEND_CPU = "CPU";
const String BACKEND_VULKAN = "Vulkan";
const String BACKEND_METAL = "Metal";
const String BACKEND_OPENCL = "OpenCL";

/// Adds native backend indices only where device labels would be ambiguous.
List<String> disambiguateBackendNames(List<String> names) {
  final counts = <String, int>{};
  for (final name in names) {
    counts[name] = (counts[name] ?? 0) + 1;
  }
  return [
    for (int i = 0; i < names.length; ++i)
      counts[names[i]]! > 1 ? '${names[i]} [$i]' : names[i],
  ];
}

String _ailiaCommonGetLlmPath() {
  if (Platform.isAndroid || Platform.isLinux) {
    return 'libailia_llm.so';
  }
  if (Platform.isMacOS) {
    return 'libailia_llm.dylib';
  }
  if (Platform.isWindows) {
    return 'ailia_llm.dll';
  }
  return 'internal';
}

DynamicLibrary _ailiaCommonGetLibrary(String path) {
  final DynamicLibrary library;
  if (Platform.isIOS) {
    library = DynamicLibrary.process();
  } else {
    library = DynamicLibrary.open(path);
  }
  return library;
}

typedef VkEnumerateInstanceVersionNative = Int32 Function(
    Pointer<Uint32> apiVersion);
typedef VkEnumerateInstanceVersionDart = int Function(
    Pointer<Uint32> apiVersion);

class AiliaLLMModel {
  static List<String> _backend = List<String>.empty();
  static List<String> _backendTypes = List<String>.empty();
  static DynamicLibrary? _sharedLibrary;
  static ailia_llm_dart.ailiaLlmFFI? _sharedApi;

  static ailia_llm_dart.ailiaLlmFFI _getApi() {
    _sharedLibrary ??= _ailiaCommonGetLibrary(_ailiaCommonGetLlmPath());
    return _sharedApi ??= ailia_llm_dart.ailiaLlmFFI(_sharedLibrary!);
  }

  Pointer<Pointer<ailia_llm_dart.AILIALLM>> pLLm = nullptr;
  dynamic dllHandle;
  bool _contextFull = false;
  Uint8List _buf = Uint8List(0);
  String _beforeText = "";
  bool _multimodalProjectorOpened = false;

  AiliaLLMModel() {}

  /// Detail of the last failed native call on this model.
  String getErrorDetail() {
    if (pLLm == nullptr || pLLm.value == nullptr) return '';
    final detail = dllHandle.ailiaLLMGetErrorDetail(pLLm.value) as Pointer<Char>;
    return detail == nullptr ? '' : detail.cast<Utf8>().toDartString();
  }

  /// Returns the device artifact stem (for example, `sm8475`) without opening
  /// a model. Throws if QNN is unavailable or the device is unsupported.
  static String getQNNModelName() {
    final api = _getApi();
    final output = calloc<Pointer<Char>>();
    try {
      final status = api.ailiaLLMGetQNNModelName(output);
      if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
        throw StateError('Failed to get QNN model name. Status: $status');
      }
      if (output.value == nullptr) {
        throw StateError('QNN model name is null');
      }
      return output.value.cast<Utf8>().toDartString();
    } finally {
      // The returned string belongs to the library; copy it before freeing output.
      calloc.free(output);
    }
  }

  static bool checkVulkanVersion() {
    try {
      final DynamicLibrary vulkanLib = Platform.isWindows
          ? DynamicLibrary.open('vulkan-1.dll')
          : DynamicLibrary.open('libvulkan.so');
      final VkEnumerateInstanceVersionDart vkEnumerateInstanceVersion =
          vulkanLib.lookupFunction<VkEnumerateInstanceVersionNative,
              VkEnumerateInstanceVersionDart>('vkEnumerateInstanceVersion');
      final Pointer<Uint32> apiVersion = calloc<Uint32>();
      final int result = vkEnumerateInstanceVersion(apiVersion);
      bool available = false;
      if (result == 0) {
        final int version = apiVersion.value;
        final int variant = (version >> 29);
        final int major = (version >> 22) & 0x7F;
        final int minor = (version >> 12) & 0x3FF;
        available = variant == 0 && (major > 1 || (major == 1 && minor >= 1));
        //print("Vulkan version ${major}.${minor}");
      }
      calloc.free(apiVersion);
      return available;
    } on Exception {
    } on ArgumentError {}
    return false;
  }

  static List<String> getBackendList() {
    if (_backend.isNotEmpty) {
      return List<String>.unmodifiable(_backend);
    }
    final api = _getApi();
    final count = calloc<UnsignedInt>();
    try {
      final status = api.ailiaLLMGetBackendCount(count);
      if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
        throw Exception('ailiaLLMGetBackendCount returned $status');
      }
      final names = <String>[];
      final types = <String>[];
      final name = calloc<Pointer<Char>>();
      try {
        for (int i = 0; i < count.value; ++i) {
          final status = api.ailiaLLMGetBackendName(name, i);
          if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS ||
              name.value == nullptr) {
            throw Exception('ailiaLLMGetBackendName returned $status');
          }
          final nativeName = name.value.cast<Utf8>().toDartString();
          // ggml calls its Metal registry "MTL"; keep Flutter's public name.
          final type = nativeName == 'MTL' ? BACKEND_METAL : nativeName;
          types.add(type);
          final deviceStatus = api.ailiaLLMGetBackendDeviceName(name, i);
          if (deviceStatus != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS ||
              name.value == nullptr) {
            throw Exception('ailiaLLMGetBackendDeviceName returned $deviceStatus');
          }
          final deviceName = name.value.cast<Utf8>().toDartString();
          names.add(type == BACKEND_CPU ? type : '$type: $deviceName');
        }
      } finally {
        calloc.free(name);
      }
      _backend = disambiguateBackendNames(names);
      _backendTypes = types;
      return List<String>.unmodifiable(_backend);
    } finally {
      calloc.free(count);
    }
  }

  /// Opens a GGUF model or a self-contained, SoC-specific QNN text package.
  /// With no backend argument, GGUF uses automatic GPU fitting with CPU
  /// fallback, while .qnn selects HTP (QNN). Explicit Metal/GPU selection
  /// still fits layers to available memory, but requires some GPU offload.
  /// CPU/GPU for .qnn or HTP for GGUF is rejected.
  /// Use [openMultimodalProjectorFile] afterwards for vision or audio.
  /// A context size of zero selects the package/model default.
  void open(String modelPath, int nCtx, {String backend = ""}) {
    if (pLLm != nullptr) {
      close();
    }

    // Reset multimodal projector state when opening a new model
    _multimodalProjectorOpened = false;

    final List<String> backendList = getBackendList();
    if (backendList.isEmpty) {
      throw Exception('ailiaLLM no available backend found');
    }
    // The native API chooses GGUF GPU/CPU placement or QNN HTP by model format
    // when the caller did not explicitly select a backend.
    final automatic = backend.isEmpty;
    var backendIdx = automatic ? -1 : backendList.indexOf(backend);
    if (!automatic && backendIdx < 0) {
      backendIdx = _backendTypes.indexOf(backend);
    }
    if (!automatic && backendIdx < 0) {
      throw Exception('ailiaLLM backend not found: $backend');
    }
    dllHandle = _getApi();

    pLLm = malloc<Pointer<ailia_llm_dart.AILIALLM>>();
    pLLm.value = nullptr;

    var status = dllHandle.ailiaLLMCreate(pLLm);
    if (status != 0) {
      close();
      throw Exception("ailiaLLMCreate returned an error status $status");
    }

    if (!automatic) {
      status = dllHandle.ailiaLLMSetBackend(pLLm.value, backendIdx);
      if (status != 0) {
        close();
        throw Exception('ailiaLLMSetBackend returned an error status $status');
      }
    }

    if (Platform.isWindows) {
      Pointer<WChar> path = modelPath.toNativeUtf16().cast<WChar>();
      status = dllHandle.ailiaLLMOpenModelFileW(pLLm.value, path, nCtx);
      malloc.free(path);
    } else {
      Pointer<Char> path = modelPath.toNativeUtf8().cast<Char>();
      status = dllHandle.ailiaLLMOpenModelFileA(pLLm.value, path, nCtx);
      malloc.free(path);
    }
    if (status != 0) {
      close();
      throw Exception("ailiaLLMOpenModelFile returned an error status $status");
    }
  }

  /// Free memory allocated natively.
  void close() {
    if (pLLm != nullptr) {
      if (pLLm.value != nullptr) {
        dllHandle.ailiaLLMDestroy(pLLm.value);
        pLLm.value = nullptr;
      }
      malloc.free(pLLm);
      pLLm = nullptr;
    }
    // Reset multimodal projector state when closing the model
    _multimodalProjectorOpened = false;
  }

  void setSamplingParams(int top_k, double top_p, double temp, int dist) {
    if (pLLm == nullptr) {
      throw Exception("ailia LLM not initialized.");
    }

    var status = dllHandle.ailiaLLMSetSamplingParams(
        pLLm.value, top_k, top_p, temp, dist);
    if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
      throw Exception("ailiaLLMGenerate returned an error status $status");
    }
  }

  /// Enable or disable thinking (reasoning output).
  /// Controls whether thinking models (e.g. Gemma4) output their reasoning process.
  /// Must be called before setPrompt.
  void setThinking(bool enable) {
    if (pLLm == nullptr) {
      throw Exception("ailia LLM not initialized.");
    }

    var status = dllHandle.ailiaLLMSetThinking(pLLm.value, enable ? 1 : 0);
    if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
      throw Exception("ailiaLLMSetThinking returned an error status $status");
    }
  }

  /// Check if any message in the list contains media_data.
  bool _hasMediaData(List<Map<String, dynamic>> messages) {
    for (var message in messages) {
      if (message.containsKey('media_data') && message['media_data'] != null) {
        final mediaData = message['media_data'];
        if (mediaData is List && mediaData.isNotEmpty) {
          return true;
        }
      }
    }
    return false;
  }

  /// Set the tool (function) definitions for tool use (function calling).
  ///
  /// [tools] is an OpenAI-compatible list of tool definitions, e.g.
  /// ```dart
  /// llm.setTools([
  ///   {
  ///     'type': 'function',
  ///     'function': {
  ///       'name': 'get_weather',
  ///       'description': 'Get the current weather',
  ///       'parameters': {
  ///         'type': 'object',
  ///         'properties': {'city': {'type': 'string'}},
  ///         'required': ['city'],
  ///       },
  ///     },
  ///   },
  /// ]);
  /// ```
  /// Pass null or an empty list to clear the tools.
  ///
  /// The tools are rendered into the prompt through the chat template on the
  /// next [setPromptJson] call, the output is constrained to the tool call syntax,
  /// and the buffered output can be retrieved with [getResponseJson].
  ///
  /// While tools are set, setPrompt fails with INVALID_STATE. Use setPromptJson
  /// and getResponseJson. Deltas remain available for streaming previews.
  ///
  /// Available for models whose chat template supports tool calling (e.g. Gemma 4).
  void setTools(List<Map<String, dynamic>>? tools) {
    if (pLLm == nullptr) {
      throw Exception("ailia LLM not initialized.");
    }

    int status;
    if (tools == null || tools.isEmpty) {
      status = dllHandle.ailiaLLMSetTools(pLLm.value, nullptr);
    } else {
      Pointer<Char> toolsJson = jsonEncode(tools).toNativeUtf8().cast<Char>();
      status = dllHandle.ailiaLLMSetTools(pLLm.value, toolsJson);
      malloc.free(toolsJson);
    }
    if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
      throw Exception("ailiaLLMSetTools returned an error status $status");
    }
  }

  /// Convert the content of a message to the string passed to the native API.
  /// For role 'tool' a Map content is serialized as JSON.
  String _messageContent(Map<String, dynamic> message) {
    final content = message['content'];
    if (message['role'] == 'tool' && content is! String) {
      return jsonEncode(content);
    }
    return content as String;
  }

  /// Sets structured JSON history, required with tools. User content arrays
  /// support text and image or audio with file_path or base64 data. Load a
  /// matching projector first; submit image and audio in separate prompts.
  void setPromptJson(List<Map<String, dynamic>> messages) {
    if (pLLm == nullptr) throw Exception("ailia LLM not initialized.");
    final text = jsonEncode(messages).toNativeUtf8();
    try {
      final status = dllHandle.ailiaLLMSetPromptJson(pLLm.value, text.cast<Char>());
      _contextFull = status == ailia_llm_dart.AILIA_LLM_STATUS_CONTEXT_FULL;
      if (status != 0) throw Exception("SetPromptJson failed: $status");
      _buf = Uint8List(0);
      _beforeText = "";
    } finally { malloc.free(text); }
  }

  /// Gets buffered assistant JSON after generation; no delta concatenation needed.
  Map<String, dynamic> getResponseJson() {
    if (pLLm == nullptr) throw Exception("ailia LLM not initialized.");
    final size = calloc<UnsignedInt>();
    try {
      int status = dllHandle.ailiaLLMGetResponseJsonSize(pLLm.value, size);
      if (status != 0) throw Exception("GetResponseJsonSize failed: $status");
      final output = malloc<Char>(size.value);
      try {
        status = dllHandle.ailiaLLMGetResponseJson(pLLm.value, output, size.value);
        if (status != 0) throw Exception("GetResponseJson failed: $status");
        return jsonDecode(output.cast<Utf8>().toDartString()) as Map<String, dynamic>;
      } finally { malloc.free(output); }
    } finally { calloc.free(size); }
  }

  void setPrompt(List<Map<String, dynamic>> messages) {
    if (pLLm == nullptr) {
      throw Exception("ailia LLM not initialized.");
    }

    bool hasMedia = _hasMediaData(messages);

    // If media_data exists, check that the multimodal projector is loaded
    if (hasMedia) {
      if (!_multimodalProjectorOpened) {
        throw Exception(
            "media_data was provided but multimodal projector is not loaded. "
            "Call openMultimodalProjectorFile() first to enable multimodal generation.");
      }
      // Use multimodal path
      _setMultimodalPromptInternal(messages);
    } else {
      // Use text-only path
      _setTextPromptInternal(messages);
    }
  }

  /// Internal implementation for text-only prompts.
  void _setTextPromptInternal(List<Map<String, dynamic>> messages) {
    // Allocate an array of ailia_llm_chat_message_t and initialize it
    // with the messages data.
    final messagesPtr =
        calloc<ailia_llm_dart.AILIALLMChatMessage>(messages.length);

    try {
      for (var i = 0; i < messages.length; i++) {
        if (!messages[i].containsKey("content")) {
          throw Exception("missing 'content' property");
        }
        if (!messages[i].containsKey("role")) {
          throw Exception("missing 'role' property");
        }

        final content = _messageContent(messages[i]);
        final role = messages[i]['role'] as String;
        final p = messagesPtr[i];

        p.content = content.toNativeUtf8().cast<Char>();
        p.role = role.toNativeUtf8().cast<Char>();
      }

      _contextFull = false;
      _buf = Uint8List(0);
      _beforeText = "";

      int status =
          dllHandle.ailiaLLMSetPrompt(pLLm.value, messagesPtr, messages.length);
      if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
        if (status == ailia_llm_dart.AILIA_LLM_STATUS_CONTEXT_FULL) {
          _contextFull = true;
          return;
        }
        throw Exception("ailiaLLMSetPrompt returned an error status $status");
      }
    } finally {
      // free string
      for (var i = 0; i < messages.length; i++) {
        final p = messagesPtr[i];
        if (p.content != nullptr) {
          malloc.free(p.content);
        }
        if (p.role != nullptr) {
          malloc.free(p.role);
        }
      }
      malloc.free(messagesPtr);
    }
  }

  /// Ask the model to generate the next token.
  /// This function properly handle incomplete multi-byte utf8 character.
  String? generate() {
    if (pLLm == nullptr) {
      throw Exception("ailia LLM not initialized.");
    }

    Pointer<Uint32> done = malloc<Uint32>();
    var status = dllHandle.ailiaLLMGenerate(
      pLLm.value,
      done,
    );
    int doneFlag = done.value;
    malloc.free(done);

    _contextFull = false;

    if (doneFlag == 1) {
      return null;
    }

    if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
      if (status == ailia_llm_dart.AILIA_LLM_STATUS_CONTEXT_FULL) {
        _contextFull = true;
        return null;
      }
      throw Exception("ailiaLLMGenerate returned an error status $status");
    }

    // Try first with gBuff which is a buffer associated to this
    // prompt instance.
    final Pointer<UnsignedInt> size = malloc<UnsignedInt>();
    dllHandle.ailiaLLMGetDeltaTextSize(pLLm.value, size);

    final Pointer<Char> byteBuffer = malloc<Char>(size.value);
    dllHandle.ailiaLLMGetDeltaText(pLLm.value, byteBuffer, size.value);

    var buffer = Uint8List(size.value - 1);
    for (var i = 0; i < size.value - 1; i++) {
      buffer[i] = byteBuffer.elementAt(i).value;
    }

    Uint8List combinedUint8List = Uint8List(_buf.length + buffer.length);
    combinedUint8List.setRange(0, _buf.length, _buf);
    combinedUint8List.setRange(
        _buf.length, _buf.length + buffer.length, buffer);
    _buf = combinedUint8List;

    malloc.free(size);
    malloc.free(byteBuffer);

    String deltaText = "";
    try {
      String text = utf8.decode(_buf);
      if (_beforeText.length != text.length) {
        deltaText = text.substring(_beforeText.length);
      }
      _beforeText = text;
    } on FormatException catch (e) {
      // unicode decode error
    }

    return deltaText;
  }

  bool contextFull() {
    return _contextFull;
  }

  // Get token count
  int getTokenCount(String text) {
    if (pLLm == nullptr) {
      throw Exception("ailia LLM not initialized.");
    }

    final Pointer<UnsignedInt> count = malloc<UnsignedInt>();
    Pointer<Char> pText = text.toNativeUtf8().cast<Char>();
    dllHandle.ailiaLLMGetTokenCount(pLLm.value, count, pText);
    int retCount = count.value;
    malloc.free(count);
    return retCount;
  }

  /// Opens a matching vision/audio projector after [open]. Accepts a GGUF
  /// projector or a self-contained, SoC-specific QNN projector package.
  void openMultimodalProjectorFile(String mmprojPath) {
    if (pLLm == nullptr) {
      throw Exception("ailia LLM not initialized.");
    }

    int status;
    if (Platform.isWindows) {
      Pointer<WChar> path = mmprojPath.toNativeUtf16().cast<WChar>();
      status =
          dllHandle.ailiaLLMOpenMultimodalProjectorFileW(pLLm.value, path);
      malloc.free(path);
    } else {
      Pointer<Char> path = mmprojPath.toNativeUtf8().cast<Char>();
      status =
          dllHandle.ailiaLLMOpenMultimodalProjectorFileA(pLLm.value, path);
      malloc.free(path);
    }
    if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
      throw Exception(
          "ailiaLLMOpenMultimodalProjectorFile returned an error status $status");
    }
    _multimodalProjectorOpened = true;
  }

  /// Returns `vision` and `audio` support flags for the opened projector.
  Map<String, bool> getMultimodalCapabilities() {
    if (pLLm == nullptr) {
      throw Exception("ailia LLM not initialized.");
    }

    final Pointer<UnsignedInt> visionSupport = malloc<UnsignedInt>();
    final Pointer<UnsignedInt> audioSupport = malloc<UnsignedInt>();

    int status = dllHandle.ailiaLLMGetMultimodalCapabilities(
        pLLm.value, visionSupport, audioSupport);

    bool vision = visionSupport.value != 0;
    bool audio = audioSupport.value != 0;

    malloc.free(visionSupport);
    malloc.free(audioSupport);

    if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
      throw Exception(
          "ailiaLLMGetMultimodalCapabilities returned an error status $status");
    }

    return {"vision": vision, "audio": audio};
  }

  /// Internal implementation for multimodal prompts.
  void _setMultimodalPromptInternal(List<Map<String, dynamic>> messages) {
    // Allocate an array of AILIALLMMultimodalChatMessage and initialize it
    final messagesPtr =
        calloc<ailia_llm_dart.AILIALLMMultimodalChatMessage>(messages.length);

    try {
      for (var i = 0; i < messages.length; i++) {
        if (!messages[i].containsKey("content")) {
          throw Exception("missing 'content' property");
        }
        if (!messages[i].containsKey("role")) {
          throw Exception("missing 'role' property");
        }

        final content = _messageContent(messages[i]);
        final role = messages[i]['role'] as String;
        final p = messagesPtr[i];

        p.content = content.toNativeUtf8().cast<Char>();
        p.role = role.toNativeUtf8().cast<Char>();

        // Handle media data if present
        if (messages[i].containsKey('media_data') &&
            messages[i]['media_data'] != null) {
          final mediaList =
              messages[i]['media_data'] as List<Map<String, dynamic>>;
          if (mediaList.isNotEmpty) {
            final mediaPtr =
                calloc<ailia_llm_dart.AILIALLMMediaData>(mediaList.length);
            p.media_data = mediaPtr;
            p.media_count = mediaList.length;

            for (var j = 0; j < mediaList.length; j++) {
              final media = mediaList[j];
              final mediaData = mediaPtr[j];

              mediaData.media_type =
                  (media['media_type'] as String).toNativeUtf8().cast<Char>();
              final filePath = media['file_path'];
              final data = media['data'];
              if ((filePath is String && filePath.isNotEmpty) ==
                  (data != null)) {
                throw ArgumentError(
                    'Exactly one of file_path or data is required for media_data');
              }
              mediaData.file_path = filePath is String
                  ? filePath.toNativeUtf8().cast<Char>()
                  : nullptr;
              if (data != null) {
                final bytes = data is Uint8List
                    ? data
                    : Uint8List.fromList((data as List).cast<int>());
                if (bytes.isEmpty) {
                  throw ArgumentError('Media data buffer must not be empty');
                }
                mediaData.data = malloc<UnsignedChar>(bytes.length);
                mediaData.data
                    .cast<Uint8>()
                    .asTypedList(bytes.length)
                    .setAll(0, bytes);
                mediaData.data_size = bytes.length;
              } else {
                mediaData.data = nullptr;
                mediaData.data_size = 0;
              }
              mediaData.width = media['width'] ?? 0;
              mediaData.height = media['height'] ?? 0;
            }
          } else {
            p.media_data = nullptr;
            p.media_count = 0;
          }
        } else {
          p.media_data = nullptr;
          p.media_count = 0;
        }
      }

      _contextFull = false;
      _buf = Uint8List(0);
      _beforeText = "";

      int status = dllHandle.ailiaLLMSetMultimodalPrompt(
          pLLm.value, messagesPtr, messages.length);
      if (status != ailia_llm_dart.AILIA_LLM_STATUS_SUCCESS) {
        if (status == ailia_llm_dart.AILIA_LLM_STATUS_CONTEXT_FULL) {
          _contextFull = true;
          return;
        }
        throw Exception(
            "ailiaLLMSetMultimodalPrompt returned an error status $status");
      }
    } finally {
      // free strings and media data
      for (var i = 0; i < messages.length; i++) {
        final p = messagesPtr[i];
        if (p.content != nullptr) {
          malloc.free(p.content);
        }
        if (p.role != nullptr) {
          malloc.free(p.role);
        }
        if (p.media_data != nullptr) {
          for (var j = 0; j < p.media_count; j++) {
            final mediaData = p.media_data[j];
            if (mediaData.media_type != nullptr) {
              malloc.free(mediaData.media_type);
            }
            if (mediaData.file_path != nullptr) {
              malloc.free(mediaData.file_path);
            }
            if (mediaData.data != nullptr) {
              malloc.free(mediaData.data);
            }
          }
          malloc.free(p.media_data);
        }
      }
      malloc.free(messagesPtr);
    }
  }

  /// Set multimodal prompt for generation with media attachments.
  ///
  /// @deprecated Use [setPrompt] instead. This method is deprecated and will
  /// be removed in a future version. The unified [setPrompt] method
  /// automatically detects media_data in messages and routes accordingly.
  ///
  /// messages must be a list of maps with the following properties:
  /// - 'role' (String): The role (e.g., "system", "user", "assistant")
  /// - 'content' (String): The text content with <__media__> placeholders
  /// - 'media_data' (List<Map<String, dynamic>>, optional): Media attachments
  @Deprecated(
      'Use setPrompt() instead, which automatically detects media_data in messages.')
  void setMultimodalPrompt(List<Map<String, dynamic>> messages) {
    // Delegate to the unified setPrompt() method to ensure consistent behavior
    // and projector-loaded checks.
    setPrompt(messages);
  }
}
