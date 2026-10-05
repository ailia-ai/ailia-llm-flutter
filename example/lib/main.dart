import 'package:flutter/material.dart';
import 'dart:async';

import 'package:ailia_llm/ailia_llm_model.dart';
import 'utils/download_model.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final AiliaLLMModel _ailiaLlmModel = AiliaLLMModel();
  String _predictText = "Plese push plus button";
  String _backend = "";
  String _profileText = "";

  @override
  void initState() {
    super.initState();
    initPlatformState();
  }

  // Platform messages are asynchronous, so we initialize in an async method.
  Future<void> initPlatformState() async {}

  String _concatMessage(List<Map<String, dynamic>> messages) {
    String text = "";
    for (int i = 0; i < messages.length; i++) {
      text = text + messages[i]["role"] + " : " + messages[i]["content"] + "\n";
    }
    return text;
  }

  void _ailiaLLMTest() async {
    setState(() {
      _predictText = "Downloading...";
    });

    downloadModel(
        "https://storage.googleapis.com/ailia-models/gemma/gemma-4-E2B-it-Q4_K_M.gguf",
        "gemma-4-E2B-it-Q4_K_M.gguf", (model_file) {
      setState(() {
        _predictText = "Processing...";
      });

      int nCtx = 8192;
      _ailiaLlmModel.open(model_file.path, nCtx, backend: _backend);

      int startTime = DateTime.now().millisecondsSinceEpoch;

      List<Map<String, dynamic>> messages =
          List<Map<String, dynamic>>.empty(growable: true);
      messages.add({"role": "system", "content": "語尾に「わん」をつけてください。"});
      messages.add({"role": "user", "content": "こんにちは。"});

      _ailiaLlmModel.setPrompt(messages);
      _predictText = "";
      while (true) {
        String? deltaText = _ailiaLlmModel.generate();
        if (deltaText == null) {
          break;
        }
        _predictText = _predictText + deltaText;
      }

      messages.add({"role": "assistant", "content": _predictText});

      setState(() {
        _predictText = _concatMessage(messages);
      });

      messages.add({"role": "user", "content": "前回の回答を英語にしてください。"});

      _ailiaLlmModel.setPrompt(messages);
      _predictText = "";
      while (true) {
        String? deltaText = _ailiaLlmModel.generate();
        if (deltaText == null) {
          break;
        }
        _predictText = _predictText + deltaText;
      }

      messages.add({"role": "assistant", "content": _predictText});

      if (_ailiaLlmModel.contextFull()) {
        print("Context Full");
      }

      int contextSize = _ailiaLlmModel.getTokenCount("トークンの数を数えます。");
      if (contextSize != 7) {
        _predictText = "contextSize error $contextSize";
      }

      int endTime = DateTime.now().millisecondsSinceEpoch;
      _profileText = "processing time : ${(endTime - startTime) / 1000} sec";

      _ailiaLlmModel.close();

      setState(() {
        _predictText = _concatMessage(messages);
        _profileText = _profileText;
      });
    });
  }

  void _incrementCounter() async {
    _ailiaLLMTest();
  }

  @override
  Widget build(BuildContext context) {
    List<String> backendList = AiliaLLMModel.getBackendList();
    if (_backend == "") {
      _backend = backendList[0];
    }
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('ailia LLM example app'),
        ),
        body: Center(
          child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: <Widget>[
                Text('$_predictText\n'),
                Text('$_profileText\n'),
                DropdownButton(
                  items: backendList
                      .map((item) => DropdownMenuItem(
                            child: Text(item),
                            value: item,
                          ))
                      .toList(),
                  onChanged: (String? value) {
                    setState(() {
                      _backend = value!;
                    });
                  },
                  value: _backend,
                ),
              ]),
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: _incrementCounter,
          tooltip: 'Inference',
          child: const Icon(Icons.add),
        ),
      ),
    );
  }
}
