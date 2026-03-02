from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from core.response import R
from core.logger import logger


def register_exception(app):
    """注册全局异常处理"""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(f"HTTP异常: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content=R.fail(msg=str(exc.detail), code=exc.status_code).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"参数验证失败: {exc.errors()}")
        return JSONResponse(content=R.fail(msg=f"参数验证失败",data=exc.errors()).model_dump())
    @app.exception_handler(AssertionError)
    async def assertion_exception_handler(request: Request, exc: AssertionError):
        logger.warning("参数校验失败: %s", exc)
        return JSONResponse(content=R.fail(msg=str(exc)).model_dump()
    )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("服务器内部错误", exc_info=exc)
        return JSONResponse(content=R.fail(msg="服务器内部错误").model_dump())

    logger.info("全局异常处理已注册")
