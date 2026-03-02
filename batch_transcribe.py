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

# 音频文件目录
data_dir = "./datasets"

# 支持的音频扩展名
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# 模型文件路径
model_dir = "./model"
encoder_onnx_path = f"{model_dir}/Fun-ASR-Nano-Encoder-Adaptor.fp16.onnx"
ctc_onnx_path = f"{model_dir}/Fun-ASR-Nano-CTC.fp16.onnx"
decoder_gguf_path = f"{model_dir}/Fun-ASR-Nano-Decoder.fp16.gguf"
tokens_path = f"{model_dir}/tokens.txt"
hotwords_path = "./hot-word.txt"

# 语言设置（None=自动检测）
language = None

# 上下文信息（留空则不使用）
context = (
    "这是民航空中交通管制（ATC）地空通话录音，通话语言为中英文混合。"
    "说话人为管制员或飞行员，内容涉及起飞、落地、滑行、航向、高度、速度等指令。"
    "中文数字采用航空专用读法：0=洞，1=幺，2=两，3=三，4=四，5=五，6=六，7=拐，8=八，9=勾。"
    "英文部分使用ICAO标准术语，如CLEARED、MAINTAIN、DESCEND、CLIMB、RUNWAY、SQUAWK，"
    "以及NATO音标字母如ALFA、BRAVO、CHARLIE等。"
    "请严格保留原始口语读法，中文数字不要转为阿拉伯数字，英文术语保留原词。"
    "常见句式如：南方三五六幺请联系进近幺两幺点六五、CCA1508 DESCEND AND MAINTAIN FL090、修正海压幺洞幺三。"
    "通话中可能包含航班呼号（如国航、东方、南方、Cathay、Lufthansa）加数字编号、"
    "跑道编号、频率读报、应答机编码、高度层和航向角度。"
)

# 是否启用 CTC 辅助
enable_ctc = True

# 输出结果文件
output_json = "transcribe_results.json"

# ==================== 主逻辑 ====================


def collect_audio_files(directory):
    """收集目录中所有音频文件，按文件名排序"""
    audio_files = []
    for filename in sorted(os.listdir(directory)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in AUDIO_EXTENSIONS:
            audio_files.append(filename)
    return audio_files


def main():
    print("=" * 70)
    print("批量 ASR 转写")
    print("=" * 70)

    # 收集音频文件
    audio_files = collect_audio_files(data_dir)
    if not audio_files:
        print(f"在 {data_dir} 中未找到音频文件。")
        return 1

    print(f"\n找到 {len(audio_files)} 个音频文件：")
    for f in audio_files:
        print(f"  - {f}")

    # 创建 ASR 引擎
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
    engine.transcribe(first_file, language=language, context=context, verbose=False, duration=5.0)

    # 逐个转写
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
                temperature=0.2,
            )
            cost = time.time() - start_time

            # 提取纯文本
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            results[filename] = text

            print(f"  耗时: {cost:.1f}s")
            print(f"  结果: {text[:200]}{'...' if len(text) > 200 else ''}")
            print()

        except Exception as e:
            cost = time.time() - start_time
            print(f"  ✗ 失败 ({cost:.1f}s): {e}")
            results[filename] = f"[ERROR] {e}" 
            print()

    # 保存结果到 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print("=" * 70)
    print("转写完成！结果汇总：")
    print("=" * 70)
    for filename, text in results.items():
        print(f"\n📄 {filename}")
        print(f"   {text}")

    print(f"\n结果已保存到: {output_json}")

    # 清理资源
    engine.cleanup()
    return 0


if __name__ == "__main__":
    exit(main())
