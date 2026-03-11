import asyncio
from concurrent.futures import ThreadPoolExecutor
from core.config import settings
from fun_asr_gguf import FunASREngine, ASREngineConfig
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
        config = ASREngineConfig(
            encoder_onnx_path=settings.encoder_onnx_path,
            ctc_onnx_path=settings.ctc_onnx_path,
            decoder_gguf_path=settings.decoder_gguf_path,
            tokens_path=settings.tokens_path,
            hotwords_path=settings.HOTWORDS_PATH,
            similar_threshold=settings.SIMILAR_THRESHOLD,
            max_hotwords=settings.MAX_HOTWORDS,
            enable_ctc=settings.ENABLE_CTC,
            onnx_provider="cuda",
            verbose=False,
        )
        self.engine = FunASREngine(config)
        logger.info("ASR 引擎初始化成功。")

    def cleanup(self):
        if self.engine:
            logger.info("正在清理 ASR 引擎...")
            self.engine.cleanup()
            self.engine = None
        self._executor.shutdown(wait=True)

    def _transcribe_sync(
        self,
        filepath: str,
        language: str = None,
        context: str = None,
        temperature: float = 0.0,
    ) -> dict:
        if not self.engine:
            raise RuntimeError("ASR 引擎尚未初始化")

        result = self.engine.transcribe(
            filepath,
            language=language,
            context=context,
            verbose=False,
            segment_size=60.0,
            overlap=4.0,
            start_second=0.0,
            duration=None,
            temperature=temperature,
        )

        # 关键修复：清理decoder状态，防止单例引擎内部状态污染
        # 问题根源：06-Inference.py每次运行新建->初始化->预热->转写->cleanup
        #         web service使用单例，长期运行，从未cleanup，导致内部状态污染
        # 解决方案：每次转写后，重置decoder和模型上下文，模拟cleanup的效果
        try:
            if self.engine.orchestrator and hasattr(
                self.engine.orchestrator, "decoder"
            ):
                # 重新创建decoder，清理KV缓存和生成状态
                from fun_asr_gguf.inference.core.decoder import StreamDecoder

                self.engine.orchestrator.decoder = StreamDecoder(self.engine.models)

            # 显式清理模型的LLM上下文KV缓存
            if self.engine.models and self.engine.models.ctx:
                self.engine.models.ctx.clear_kv_cache()

            logger.debug("已清理decoder和KV缓存")
        except Exception as e:
            logger.warning(f"状态清理失败（继续处理）: {e}")

        if isinstance(result, dict):
            return result
        import dataclasses

        if dataclasses.is_dataclass(result):
            return dataclasses.asdict(result)
        return {"text": str(result)}

    async def transcribe_async(
        self,
        filepath: str,
        language: str = None,
        context: str = None,
        temperature: float = 0.0,
    ) -> dict:
        async with self._lock:
            loop = asyncio.get_running_loop()
            result_dict = await loop.run_in_executor(
                self._executor,
                self._transcribe_sync,
                filepath,
                language,
                context,
                temperature,
            )
            return result_dict

    def get_hotwords(self) -> list:
        """获取当前配置的热词列表"""
        if not self.engine or getattr(self.engine, "models", None) is None:
            return []
        if getattr(self.engine.models, "hotword_manager", None) is None:
            return []

        # 从字典键中获取当前的所有热词, 位于 phoneme_corrector 中
        try:
            hw_dict = self.engine.models.hotword_manager.phoneme_corrector.hotwords
            return list(hw_dict.keys()) if hw_dict else []
        except AttributeError:
            return []

    def update_hotwords(self, words: list) -> int:
        """更新热词文件并触发内存重载"""
        # 将列表写入文件
        content = "\n".join(words)

        path = settings.HOTWORDS_PATH
        if not path:
            raise ValueError("热词文件路径未配置")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")

        # 触发立即重载
        if (
            self.engine
            and getattr(self.engine, "models", None)
            and getattr(self.engine.models, "hotword_manager", None)
        ):
            self.engine.models.hotword_manager._load_hot()
            try:
                return len(
                    self.engine.models.hotword_manager.phoneme_corrector.hotwords
                )
            except AttributeError:
                pass

        return len(words)


asr_service = ASRService()
