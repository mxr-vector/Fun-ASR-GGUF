import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import args
from core.logger import logger
from core.gobal_exception import register_exception
from core.middleware_access_log import AccessLogMiddleware
from core.middleware_auth import TokenAuthMiddleware
from core.middleware_request_id import RequestIDMiddleware
from core.auto_import import load_routers
from services import asr_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：初始化 ASR 引擎
    logger.info("正在启动 FastAPI 应用程序...")
    try:
        asr_service.initialize()
    except Exception as e:
        logger.error(f"ASR 引擎初始化失败: {e}", exc_info=True)
        raise e
        
    yield
    
    # 关闭时：清理 ASR 引擎和线程池
    logger.info("正在关闭 FastAPI 应用程序...")
    asr_service.cleanup()


app = FastAPI(
    title="Fun-ASR Fast API Service",
    description="ASR service wrapping batch_transcribe logic into a thread-safe web api.",
    version="1.0.0",
    lifespan=lifespan
)

# 可选：禁用 CORS 限制以防万一
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 核心中间件
app.add_middleware(AccessLogMiddleware)
app.add_middleware(TokenAuthMiddleware)
app.add_middleware(RequestIDMiddleware)

# 注册核心异常处理
register_exception(app)

# 自动加载 routers 包下的路由
load_routers(app, package="routers")





if __name__ == "__main__":
    logger.info(f"服务启动于 {args.host}:{args.port}")
    uvicorn.run(
        "infer:app", 
        host=args.host, 
        port=args.port, 
        reload=False
    )
