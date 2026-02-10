from fun_asr_gguf import create_asr_engine

# 创建并初始化引擎 (推荐使用单例或长期持有实例)
engine = create_asr_engine(
    encoder_onnx_path="model/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx",
    ctc_onnx_path="model/Fun-ASR-Nano-CTC.int8.onnx",
    decoder_gguf_path="model/Fun-ASR-Nano-Decoder.q8_0.gguf",
    tokens_path="model/tokens.txt",
    hotwords_path="hot.txt",  # 可选：热词文件路径，支持运行期间实时修改
    similar_threshold=0.6,  # 可选：热词模糊匹配阈值，默认 0.6
    max_hotwords=10,  # 可选：最多提供给 LLM 的热词数量，默认 10
)
engine.initialize()

result = engine.transcribe("audio.mp3", language="中文")
print(result.text)
