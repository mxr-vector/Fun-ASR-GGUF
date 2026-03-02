from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from core.config import args

# 可配置无需认证的路径
EXCLUDE_PATHS = {"/", "/docs", "/openapi.json", "/favicon.ico", "/static"}


class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 跳过无需认证的路径
        if path in EXCLUDE_PATHS or path.startswith("/public"):
            return await call_next(request)

        token = request.headers.get("Authorization")

        if token != f"Bearer {args.web_secret_key}":
            return JSONResponse(
                status_code=401,
                content={"code": 401, "msg": "Invalid or missing token", "data": None},
            )

        return await call_next(request)
