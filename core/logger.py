import sys
from loguru import logger
from contextvars import ContextVar
from pathlib import Path

request_id_ctx = ContextVar("request_id", default="-")

BASE_DIR = Path(__file__).resolve().parent
LOG_ROOT = BASE_DIR / "logs"

logger.remove()

# 文件

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level:<8} | "
    "{extra[request_id]} | "
    "{message}"
)
# 控制台
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level="INFO",
)

# 所有日志
logger.add(
    LOG_ROOT / "{time:YYYY-MM-DD}/app.log",
    format=LOG_FORMAT,
    level="INFO",
    rotation="100 MB",  # 单文件最大 100MB
    retention="5 days",
    compression="zip",
    enqueue=True,
)


# 错误日志单独文件
logger.add(
    LOG_ROOT / "{time:YYYY-MM-DD}/error.log",
    format=LOG_FORMAT,
    level="ERROR",
    rotation="100 MB",  # 单文件最大 100MB
    retention="7 days",
    enqueue=True,
)


# 拦截器：自动注入 request_id
def inject_request_id(record):
    record["extra"]["request_id"] = request_id_ctx.get()
    return record


logger = logger.patch(inject_request_id)
