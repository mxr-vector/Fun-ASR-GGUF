import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from core.logger import logger
from core.logger import request_id_ctx


class AccessLogMiddleware(BaseHTTPMiddleware):
    '''访问日志中间件'''
    async def dispatch(self, request: Request, call_next):
        # 记录开始时间
        start_time = time.time()

        # 获取 IP
        client_host = request.client.host if request.client else "unknown"

        # 执行请求
        try:
            response = await call_next(request)
        except Exception as exc:
            # 异常时也打日志（status=500）
            cost_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[REQ] 500 {request.method} {request.url.path} from {client_host} "
                f"cost={cost_ms:.2f}ms"
            )
            raise exc

        # 正常响应时打日志
        cost_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[REQ] {response.status_code} {request.method} {request.url.path} "
            f"from {client_host} cost={cost_ms:.2f}ms"
        )

        return response
