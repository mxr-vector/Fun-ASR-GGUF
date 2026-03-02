import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from core.logger import request_id_ctx

class RequestIDMiddleware(BaseHTTPMiddleware):
    '''请求 ID 中间件'''
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())

        # 设置到 ContextVar（自动注入日志）
        request_id_ctx.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
