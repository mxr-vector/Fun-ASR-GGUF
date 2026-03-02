import asyncio
from concurrent.futures import ThreadPoolExecutor
from config import settings
from fun_asr_gguf import create_asr_engine
from core.logger import logger

class ASRService:
    def __init__(self):
        self.engine = None
        
        # --- 生产级并发与线程安全处理规则 ---
        # `fun_asr_gguf` 中的 llama.cpp / ONNX 模型和 PyTorch 实例
        # 占用大量内存，并且在执行推理时保持内部状态。
        #
        # 高并发设计：
        # 为了避免当多个请求到达 FastAPI 时，在内存中并行多个完整模型实例导致内存溢出 (OOM) 风险，
        # 我们通过一个具有 1 个工作线程的线程池对引擎操作进行串行化。
        #
        # 1. `ThreadPoolExecutor(max_workers=1)` 确保阻塞的转写函数顺序运行，这意味着同一时间只有一个请求会访问底层的 C++ 框架。
        # 2. `asyncio.Lock()` 确保重叠的 HTTP 请求能够高效地等待处理，而不会阻塞 FastAPI 的全局异步事件循环。
        #    因此，FastAPI 可以在安全地逐个处理引擎推理请求的同时，接收 100 个需要解析令牌的 HTTP 请求。
        # -------------------------------------------------------
        # -------------------------------------------------------
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()

    def initialize(self):
        if self.engine is not None:
            return
            
        logger.info("正在初始化 ASR 引擎...")
        self.engine = create_asr_engine(
            encoder_onnx_path=settings.encoder_onnx_path,
            ctc_onnx_path=settings.ctc_onnx_path,
            decoder_gguf_path=settings.decoder_gguf_path,
            tokens_path=settings.tokens_path,
            hotwords_path=settings.HOTWORDS_PATH,
            similar_threshold=settings.SIMILAR_THRESHOLD,
            max_hotwords=settings.MAX_HOTWORDS,
            enable_ctc=settings.ENABLE_CTC,
            verbose=False,
        )
        logger.info("ASR 引擎初始化成功。")

    def cleanup(self):
        if self.engine:
            logger.info("正在清理 ASR 引擎...")
            self.engine.cleanup()
            self.engine = None
        self._executor.shutdown(wait=True)

    def _transcribe_sync(self, filepath: str, language: str = None, context: str = None) -> str:
        if not self.engine:
             raise RuntimeError("ASR 引擎尚未初始化")
            
        result = self.engine.transcribe(
            filepath,
            language=language,
            context=context,
            verbose=False,
            # 使用与 batch_transcribe 相同的默认片段参数
            segment_size=60.0,
            overlap=4.0,
            temperature=0.2,
        )
        return result.get("text", "") if isinstance(result, dict) else str(result)

    async def transcribe_async(self, filepath: str, language: str = None, context: str = None) -> str:
        async with self._lock:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                self._executor, 
                self._transcribe_sync, 
                filepath, 
                language, 
                context
            )
            return text

asr_service = ASRService()
