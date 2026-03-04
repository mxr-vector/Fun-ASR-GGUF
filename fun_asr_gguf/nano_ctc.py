import numpy as np
import base64
import os
import time
from dataclasses import dataclass

@dataclass
class Token:
    text: str
    start: float
    end: float = 0.0

def load_ctc_tokens(filename):
    """加载 CTC 词表"""
    id2token = dict()
    if not os.path.exists(filename):
        return id2token
    with open(filename, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) == 1:
                t, i = " ", parts[0]
            else:
                t, i = parts
            id2token[int(i)] = t
            
            # Pre-decode base64 here to save time during inference
            try:
                # Some tokens might rely on being decoded, do it once
                id2token[int(i)] = base64.b64decode(t).decode("utf-8")
            except:
                # If fail (not b64 or other issue), keep original or handle as needed
                # For FunASR tokens, they seem to be always b64 per decode_ctc logic
                pass
                
    return id2token

def decode_ctc(logits, id2token):
    """
    Greedy search 贪错解码。
    
    Args:
        logits: 模型输出的概率分布 (T, V) 或已经是 indices (T,)
        id2token: 词表映射 dict
    """
    t0 = time.perf_counter()
    
    # [OPTIMIZATION] 如果输入已经是 1D 数组或 (1, T)，说明是已经融合了 ArgMax 的 indices
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[0] == 1):
        indices = logits.flatten()
        t_cast = 0.0
        t_argmax = 0.0
    else:
        # [Legacy] 兼容输出原始 Logits 的情况
        # 优化策略：先转 float32，避免在 float16 上做大规模 argmax 的潜在精度问题或性能抖动
        t_s = time.perf_counter()
        logits_f32 = logits.astype(np.float32)
        t_cast = time.perf_counter() - t_s
        
        t_s = time.perf_counter()
        indices = np.argmax(logits_f32, axis=-1).flatten()
        t_argmax = time.perf_counter() - t_s
        
    t0 = time.perf_counter() # 重置计数以精确测量循环耗时
    blank_id = max(id2token.keys()) if id2token else 0
    
    frame_shift_ms = 60
    offset_ms = -240
    
    # 1. Collapse repeats
    collapsed = []
    if len(indices) > 0:
        current_id = indices[0]
        start_idx = 0
        for i in range(1, len(indices)):
            if indices[i] != current_id:
                collapsed.append((current_id, start_idx))
                current_id = indices[i]
                start_idx = i
        collapsed.append((current_id, start_idx))

    results = []

    # 总帧数对应的音频结尾时间（用于计算最后一个 token 的 end）
    total_frames = len(indices)
    audio_end_time = max((total_frames * frame_shift_ms + offset_ms) / 1000.0, 0.0)

    # 2. Filter blanks and collect valid tokens with frame indices
    valid_tokens = []
    for token_id, start in collapsed:
        if token_id == blank_id:
            continue

        token_text = id2token.get(token_id, "")
        if not token_text: continue

        t_start = max((start * frame_shift_ms + offset_ms) / 1000.0, 0.0)
        valid_tokens.append((token_text, t_start))

    # 3. Compute end time: 下一个 token 的 start 即为当前 token 的 end
    for i, (text, t_start) in enumerate(valid_tokens):
        if i + 1 < len(valid_tokens):
            t_end = valid_tokens[i + 1][1]
        else:
            t_end = audio_end_time

        results.append(Token(
            text=text,
            start=t_start,
            end=t_end
        ))
                
    full_text = "".join([r.text for r in results])
    t_loop = time.perf_counter() - t0
    
    # print(f"      [Profile] Cast: {t_cast*1000:.2f}ms, Argmax: {t_argmax*1000:.2f}ms, PyLoop: {t_loop*1000:.2f}ms")
    
    timings = {
        "cast": t_cast,
        "argmax": t_argmax,
        "loop": t_loop
    }
    return full_text, results, timings

def align_timestamps(ctc_results, llm_text):
    """
    使用 Needleman-Wunsch 算法对齐 CTC 结果和 LLM 文本
    只使用起始位置进行匹配
    """
    if not ctc_results or not llm_text:
        return []

    # 1. 展开 CTC 结果为 token 级别（只保留起始位置）
    ctc_tokens = []
    for item in ctc_results:
        text = item.text
        start = item.start
        end = item.end

        if len(text) > 0:
            # 将 token 时间范围均匀分配给每个字符
            token_dur = end - start if end > start else 0.08 * len(text)
            token_duration = token_dur / len(text)
            for i, char in enumerate(text):
                c_start = start + i * token_duration
                c_end = c_start + token_duration
                ctc_tokens.append({"token": char, "start": c_start, "end": c_end})

    llm_tokens = list(llm_text)

    n = len(ctc_tokens) + 1
    m = len(llm_tokens) + 1

    # Core DP Matrix
    score = np.zeros((n, m), dtype=np.float32)
    # trace: 1=diag, 2=up, 3=left
    trace = np.zeros((n, m), dtype=np.int8)

    gap_penalty = -1.0
    match_score = 1.0
    mismatch_score = -1.0

    # Init margins
    for i in range(n): score[i][0] = i * gap_penalty
    for j in range(m): score[0][j] = j * gap_penalty

    # Fill DP
    for i in range(1, n):
        for j in range(1, m):
            token_ctc = ctc_tokens[i-1]["token"]
            token_llm = llm_tokens[j-1]

            s_diag = score[i-1][j-1] + (match_score if token_ctc.lower() == token_llm.lower() else mismatch_score)
            s_up = score[i-1][j] + gap_penalty
            s_left = score[i][j-1] + gap_penalty

            best = max(s_diag, s_up, s_left)
            score[i][j] = best

            if best == s_diag: trace[i][j] = 1
            elif best == s_up: trace[i][j] = 2
            else: trace[i][j] = 3

    # Traceback
    llm_alignment = [None] * len(llm_tokens)
    i, j = n - 1, m - 1

    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 1:
            llm_alignment[j-1] = ctc_tokens[i-1]
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i][j] == 2):
            i -= 1
        elif j > 0 and (i == 0 or trace[i][j] == 3):
            llm_alignment[j-1] = None
            j -= 1

    # 插值填充未对齐的字符
    anchors = []
    for idx, item in enumerate(llm_alignment):
        if item is not None:
            anchors.append((idx, item["start"]))

    final_tokens = []

    def get_interpolated_start(target_idx):
        """插值计算起始位置"""
        prev_a, next_a = None, None
        for a in anchors:
            if a[0] < target_idx:
                prev_a = a
            elif a[0] > target_idx:
                next_a = a
                break

        if prev_a and next_a:
            p_idx, p_start = prev_a
            n_idx, n_start = next_a

            # 线性插值
            total_gap = n_idx - p_idx
            time_gap = n_start - p_start
            step = time_gap / total_gap

            relative_step = target_idx - p_idx
            return p_start + relative_step * step
        elif prev_a:
            return prev_a[1] + 0.05  # 向后推一点
        elif next_a:
            return max(0, next_a[1] - 0.05)  # 向前推一点
        else:
            return 0.0

    for idx, char in enumerate(llm_tokens):
        if llm_alignment[idx]:
            s = llm_alignment[idx]["start"]
            e = llm_alignment[idx].get("end", s + 0.08)
        else:
            s = get_interpolated_start(idx)
            e = s + 0.08  # 默认每字符 80ms
        final_tokens.append({"token": char, "start": s, "end": e})

    # 后处理：确保相邻 token 的 end 与下一个 token 的 start 一致
    for i in range(len(final_tokens) - 1):
        final_tokens[i]["end"] = final_tokens[i + 1]["start"]

    return final_tokens
