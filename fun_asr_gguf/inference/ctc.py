import onnxruntime
import numpy as np
import base64
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from . import logger

@dataclass
class Token:
    text: str
    start: float
    is_hotword: bool = False

class CTCTokenizer:
    """
    适配器模式：将 Nano 的 Base64 词表包装成满足 HotwordRadar 要求的接口
    """
    def __init__(self, id2token, encode_fn=None):
        self.id2token = id2token
        self.token2id = {v: k for k, v in id2token.items()}
        self._piece_size = len(id2token) if id2token else 0
        
    def get_piece_size(self):
        return self._piece_size
        
    def id_to_piece(self, i):
        return self.id2token.get(i, f"<{i}>")
        
    def encode(self, text):
        result = []
        for char in text:
            tid = self.token2id.get(char)
            if tid is not None:
                result.append(tid)
        return result

    def encode_as_pieces(self, text):
        ids = self.encode(text)
        return [self.id_to_piece(i) for i in ids]


class CTCDecoder:
    """FunASR CTC 推理与解码器 (多阶段内部流水线)"""
    def __init__(self, model_path: str, tokens_path: str, onnx_provider: str = 'CPU', dml_pad_to: int = 30, corrector: Optional[Any] = None):
        self.model_path = model_path
        self.tokens_path = tokens_path
        self.onnx_provider = onnx_provider.upper()
        self.dml_pad_to = dml_pad_to
        self.corrector = corrector
        
        self.sess = None
        self.id2token = {}
        self.input_dtype = np.float32
        self.tokenizer = None
        self.radar = None
        self.integrator = None
        
        self._initialize_session()
        self._load_tokens()
        self.warmup()

    def _initialize_session(self):
        session_opts = onnxruntime.SessionOptions()
        session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        available_providers = onnxruntime.get_available_providers()
        providers = ['CPUExecutionProvider']
        
        if self.onnx_provider in ('TENSORRT', 'TRT') and 'TensorrtExecutionProvider' in available_providers:
            providers.insert(0, ('TensorrtExecutionProvider', {
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': Path(self.model_path).parent / 'trt_cache',
            }))
        elif self.onnx_provider == 'DML' and 'DmlExecutionProvider' in available_providers:
            providers.insert(0, 'DmlExecutionProvider') 
        elif self.onnx_provider == 'CUDA' and 'CUDAExecutionProvider' in available_providers:
            providers.insert(0, 'CUDAExecutionProvider')
            
        logger.info(f"[CTC] 加载模型: {os.path.basename(self.model_path)} (Providers: {providers})")
        
        self.sess = onnxruntime.InferenceSession(
            self.model_path, 
            sess_options=session_opts, 
            providers=providers
        )
        
        in_type = self.sess.get_inputs()[0].type
        self.input_dtype = np.float16 if 'float16' in in_type else np.float32

    def _load_tokens(self):
        self.id2token = load_ctc_tokens(self.tokens_path)
        self.tokenizer = CTCTokenizer(self.id2token)
        
        self.blank_id = None
        for tid, token_text in self.id2token.items():
            clean_text = token_text.lower().strip()
            if clean_text in ("<blk>", "<blank>", "<pad>"):
                self.blank_id = tid
                break
        if self.blank_id is None:
            self.blank_id = max(self.id2token.keys()) if self.id2token else 0

    def warmup(self):
        if self.dml_pad_to <= 0:
            return
        target_t_lfr = int((self.dml_pad_to * 100 + 5) // 6) + 1
        dummy_enc = np.zeros((1, target_t_lfr, 512), dtype=self.input_dtype)
        in_name = self.sess.get_inputs()[0].name
        logger.info(f"[CTC] 正在预热 (固定形状: {self.dml_pad_to}s)...")
        self.sess.run(None, {in_name: dummy_enc})

    def decode(self, enc_output: np.ndarray, enable_ctc: bool, max_hotwords: int = 10, top_k: int = 10) -> Tuple[List[Token], List[str], Dict[str, float]]:
        t_stats = {"infer": 0.0, "decode": 0.0, "radar": 0.0, "integrate": 0.0, "hotword": 0.0}
        if not enable_ctc or self.sess is None:
            return [], [], t_stats

        t0 = time.perf_counter()
        topk_log_probs, topk_indices = self._infer(enc_output)
        t_stats["infer"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        indices_2d = topk_indices[0]
        top1_indices = indices_2d[:, 0]
        ctc_text, ctc_results = self._greedy_decode(top1_indices)
        t_stats["decode"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        topk_probs = np.exp(topk_log_probs[0])
        detected_hotwords = self._radar_scan(indices_2d, topk_probs, top1_indices, top_k=top_k)
        t_stats["radar"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        if detected_hotwords and ctc_results:
            ctc_text, ctc_results = self._integrate(ctc_results, detected_hotwords)
        t_stats["integrate"] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        hotwords = [h["text"] for h in detected_hotwords]
        if self.corrector and self.corrector.hotwords and ctc_text:
            corrected_text, extra_hotwords = self._correct(ctc_text, max_hotwords)
            hotwords = list(set(hotwords) | set(extra_hotwords))
        t_stats["hotword"] = time.perf_counter() - t0
            
        return ctc_results, hotwords, t_stats

    def _infer(self, enc_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.sess.run(None, {"enc_output": enc_output})
        return outputs[0], outputs[1]

    def _greedy_decode(self, top1_indices: np.ndarray) -> Tuple[str, List[Token]]:
        # 传入实例级 blank_id，避免每次调用重新计算
        ctc_text, ctc_results, _ = decode_ctc_indices(top1_indices, self.id2token, blank_id=self.blank_id)
        return ctc_text, ctc_results

    def _radar_scan(self, indices_2d: np.ndarray, topk_probs: np.ndarray, top1_indices: np.ndarray, top_k: int = 10) -> List[Dict]:
        if self.radar is None:
            return []
        sliced_ids = indices_2d[:, :top_k]
        sliced_probs = topk_probs[:, :top_k]
        return self.radar.scan(sliced_ids, sliced_probs, top1_indices, blank_id=self.blank_id)

    def _integrate(self, ctc_results: List[Token], detected_hotwords: List[Dict]) -> Tuple[str, List[Token]]:
        if self.integrator is None:
            return "".join(r.text for r in ctc_results), ctc_results
        integrated_list = self.integrator.integrate(
            [{"text": r.text, "start": r.start} for r in ctc_results], detected_hotwords
        )
        new_results = [Token(text=r["text"], start=r["start"], is_hotword=r.get("is_hotword", False)) for r in integrated_list]
        return "".join(r.text for r in new_results), new_results

    def _correct(self, text: str, max_hotwords: int) -> Tuple[str, List[str]]:
        res = self.corrector.correct(text, k=max_hotwords)
        candidates = set()
        for _, hw, _ in res.matchs: candidates.add(hw)
        for _, hw, _ in res.similars: candidates.add(hw)
        return res.text, list(candidates)

    def set_hotword_engine(self, corrector):
        self.corrector = corrector
        
        from .radar import HotwordRadar
        from .integrator import ResultIntegrator
        
        all_hotwords = list(corrector.hotwords.keys())
        self.radar = HotwordRadar(all_hotwords, self.tokenizer)
        self.integrator = ResultIntegrator()
        
        logger.info(f"[CTC] 已绑定热词引擎 (热词数: {len(all_hotwords)})")


# ================================================================
# 模块级工具函数
# ================================================================

def load_ctc_tokens(filename: str) -> Dict[int, str]:
    """加载 CTC 词表"""
    id2token: Dict[int, str] = {}
    if not os.path.exists(filename):
        return id2token
    with open(filename, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 1:
                t, i = " ", parts[0]
            else:
                t, i = parts
            try:
                token_text = base64.b64decode(t).decode("utf-8")
            except Exception:
                token_text = t
            id2token[int(i)] = token_text
    return id2token


def decode_ctc_indices(indices, id2token: Dict[int, str], blank_id: Optional[int] = None) -> Tuple[str, List[Token], Dict]:
    """
    Greedy search 贪心解码。

    blank_id 由外部传入（复用 CTCDecoder 实例上已计算好的值），
    避免每次调用都执行 max(id2token.keys())。
    """
    t0 = time.perf_counter()

    if blank_id is None:
        blank_id = max(id2token.keys()) if id2token else 0

    frame_shift_ms = 60

    # 1. 折叠连续重复帧，记录每段起始帧索引
    collapsed: List[Tuple[int, int]] = []
    if len(indices) > 0:
        current_id = int(indices[0])
        start_idx = 0
        for i in range(1, len(indices)):
            nid = int(indices[i])
            if nid != current_id:
                collapsed.append((current_id, start_idx))
                current_id = nid
                start_idx = i
        collapsed.append((current_id, start_idx))

    # 2. 过滤 blank，构造 Token 列表
    results: List[Token] = []
    for token_id, start in collapsed:
        if token_id == blank_id:
            continue
        token_text = id2token.get(token_id, "")
        if not token_text:
            continue
        t_start = max((start * frame_shift_ms) / 1000.0, 0.0)
        results.append(Token(text=token_text, start=t_start))

    full_text = "".join(r.text for r in results)
    return full_text, results, {"loop": time.perf_counter() - t0}


def align_timestamps(ctc_results: List[Token], llm_text: str) -> List[Dict]:
    """
    使用 Needleman-Wunsch 算法对齐 CTC 结果与 LLM 文本，输出字符级时间戳。

    【核心修复】
    CTC 贪婪解码时，若头部若干帧的帧号同为 0（模型在第 0 帧连续输出多个 token），
    ctc_chars 前几项 start 会全部堆叠为 0.0。
    若直接将这些 0.0 作为 NW 锚点，LLM 对齐字符也会继承 0.0，
    后续任何插值都无法修正。

    正确做法：NW 对齐之前，先对 ctc_chars 做单调性修复——
    找到第一个 start > 0 的位置，将 [0, first_nz) 区间内字符
    均匀分配到 [0, base_time)，确保锚点时间本身单调递增。
    """
    if not ctc_results or not llm_text:
        return []

    # frame_shift_ms=60 → 每帧 60ms，单字符默认时长与此对齐
    CHAR_DURATION = 0.06

    # ---- 1. 展开 CTC Token 为字符级序列 ----
    ctc_chars: List[Dict] = []
    for item in ctc_results:
        for i, char in enumerate(item.text):
            ctc_chars.append({"token": char, "start": item.start + i * CHAR_DURATION})

    # ---- 2. 修复 ctc_chars 头部 start=0.0 堆叠 ----
    first_nz = next((i for i, c in enumerate(ctc_chars) if c["start"] > 0.0), -1)
    if first_nz > 0:
        # 将 [0, first_nz) 均匀分配到 [0, base_time)
        base_time = ctc_chars[first_nz]["start"]
        step = base_time / first_nz
        for k in range(first_nz):
            ctc_chars[k]["start"] = k * step
    elif first_nz == -1 and ctc_chars:
        # 极端情况：所有帧均为 0，用固定步长撑开
        for k in range(len(ctc_chars)):
            ctc_chars[k]["start"] = k * CHAR_DURATION

    # ---- 3. Needleman-Wunsch DP ----
    llm_chars = list(llm_text)
    n, m = len(ctc_chars) + 1, len(llm_chars) + 1
    GAP, MATCH, MISMATCH = -1.0, 1.0, -1.0

    score = np.zeros((n, m), dtype=np.float32)
    trace = np.zeros((n, m), dtype=np.int8)  # 1=diag 2=up 3=left
    score[:, 0] = np.arange(n) * GAP
    score[0, :] = np.arange(m) * GAP

    for i in range(1, n):
        ctc_tok = ctc_chars[i - 1]["token"].lower()
        for j in range(1, m):
            s_diag = score[i-1][j-1] + (MATCH if ctc_tok == llm_chars[j-1].lower() else MISMATCH)
            s_up   = score[i-1][j] + GAP
            s_left = score[i][j-1] + GAP
            best = max(s_diag, s_up, s_left)
            score[i][j] = best
            if best == s_diag:  trace[i][j] = 1
            elif best == s_up:  trace[i][j] = 2
            else:               trace[i][j] = 3

    # ---- 4. 回溯，建立 LLM 字符 → CTC 字符的对齐映射 ----
    llm_alignment: List[Optional[Dict]] = [None] * len(llm_chars)
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 1:
            llm_alignment[j - 1] = ctc_chars[i - 1]
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or trace[i][j] == 2):
            i -= 1
        else:
            llm_alignment[j - 1] = None
            j -= 1

    # ---- 5. 收集有效锚点（时间已单调，可直接用于插值） ----
    known: List[Tuple[int, float]] = [
        (idx, item["start"])
        for idx, item in enumerate(llm_alignment)
        if item is not None
    ]

    # ---- 6. 按锚点区间线性填充所有字符的 start ----
    starts = [0.0] * len(llm_chars)
    if known:
        first_idx, first_time = known[0]
        last_idx,  last_time  = known[-1]

        # 头部 [0, first_idx)：均匀分配到 [0, first_time)
        step = first_time / first_idx if first_idx > 0 and first_time > 0 else CHAR_DURATION
        for k in range(first_idx):
            starts[k] = k * step

        # 锚点自身 + 相邻锚点之间线性插值
        for ki, (l_idx, l_time) in enumerate(known):
            starts[l_idx] = l_time
            if ki + 1 < len(known):
                r_idx, r_time = known[ki + 1]
                span = r_idx - l_idx
                for k in range(l_idx + 1, r_idx):
                    starts[k] = l_time + (k - l_idx) / span * (r_time - l_time)

        # 尾部 (last_idx, end]：向后等间距
        for k in range(last_idx + 1, len(llm_chars)):
            starts[k] = last_time + (k - last_idx) * CHAR_DURATION
    else:
        # 完全无锚点，固定步长兜底
        for k in range(len(llm_chars)):
            starts[k] = k * CHAR_DURATION

    # ---- 7. 组装输出，end = 下一字符 start，末字符 +CHAR_DURATION ----
    final_chars = [{"token": char, "start": starts[idx]} for idx, char in enumerate(llm_chars)]
    for i in range(len(final_chars) - 1):
        final_chars[i]["end"] = final_chars[i + 1]["start"]
    final_chars[-1]["end"] = final_chars[-1]["start"] + CHAR_DURATION
    return final_chars