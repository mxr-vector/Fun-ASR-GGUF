"""
批量 ASR 转写脚本
遍历 data 目录中的所有音频文件，逐个进行语音识别，
输出文件名及对应的识别文本，同时保存结果到 JSON 文件。
"""

import os
import json
import time
from fun_asr_gguf import create_asr_engine

# ==================== 配置区域 ====================

data_dir = "./datasets"  # 音频文件目录
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

model_dir = "./model"
encoder_onnx_path = f"{model_dir}/Fun-ASR-Nano-Encoder-Adaptor.fp16.onnx"
ctc_onnx_path = f"{model_dir}/Fun-ASR-Nano-CTC.fp16.onnx"
decoder_gguf_path = f"{model_dir}/Fun-ASR-Nano-Decoder.fp16.gguf"
tokens_path = f"{model_dir}/tokens.txt"
hotwords_path = "./hot-word.txt"

language = None  # None=自动检测

context = ()

enable_ctc = True
output_json = "transcribe_results.json"

# ==================== 工具函数 ====================


def collect_audio_files(directory):
    """收集目录中所有音频文件，按文件名排序"""
    return [
        f
        for f in sorted(os.listdir(directory))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]


# ==================== 主流程 ====================


def main():
    print("=" * 70)
    print("批量 ASR 转写")
    print("=" * 70)

    audio_files = collect_audio_files(data_dir)
    if not audio_files:
        print(f"在 {data_dir} 中未找到音频文件。")
        return 1

    print(f"\n找到 {len(audio_files)} 个音频文件：")
    for f in audio_files:
        print(f"  - {f}")

    print("\n正在加载模型...")
    engine = create_asr_engine(
        encoder_onnx_path=encoder_onnx_path,
        ctc_onnx_path=ctc_onnx_path,
        decoder_gguf_path=decoder_gguf_path,
        tokens_path=tokens_path,
        hotwords_path=hotwords_path,
        similar_threshold=0.6,
        max_hotwords=10,
        enable_ctc=enable_ctc,
        verbose=False,
    )

    # 预跑一遍，分配内存
    print("预跑一遍，分配内存...")
    first_file = os.path.join(data_dir, audio_files[0])
    try:
        engine.transcribe(
            first_file, language=language, context=context, verbose=False, duration=5.0
        )
    except Exception as e:
        print(f"⚠ 预跑失败: {e}")

    results = {}
    total = len(audio_files)

    print(f"\n{'=' * 70}")
    print("开始批量转写")
    print(f"{'=' * 70}\n")

    for idx, filename in enumerate(audio_files, 1):
        filepath = os.path.join(data_dir, filename)
        print(f"[{idx}/{total}] 正在处理: {filename}")

        start_time = time.time()
        try:
            result = engine.transcribe(
                filepath,
                language=language,
                context=context,
                verbose=False,
                segment_size=60.0,
                overlap=4.0,
                temperature=0.0,
            )
            cost = time.time() - start_time

            # 只保留纯文本
            text = getattr(result, "text", "") if result else ""
            results[filename] = text

            print(f"  耗时: {cost:.1f}s")
            print(f"  结果: {text}\n")

        except Exception as e:
            cost = time.time() - start_time
            print(f"  ✗ 失败 ({cost:.1f}s): {e}\n")
            results[filename] = f"[ERROR] {e}"

    # 保存 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_json}")

    engine.cleanup()
    return 0


if __name__ == "__main__":
    exit(main())
