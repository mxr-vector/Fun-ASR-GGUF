import importlib
import pkgutil
from fastapi import FastAPI, APIRouter
from core.logger import logger
from core.config import args

def load_routers(app: FastAPI, package: str = "routers"):
    """
    递归扫描 routers 包及所有子包，注册每个模块中的 router 对象
    """
    parent_router = APIRouter(prefix=args.base_url)

    # 加载包
    try:
        pkg = importlib.import_module(package)
    except ModuleNotFoundError:
        logger.error(f"未找到包: {package}")

    # 扫描当前包的所有模块和子包
    for _, module_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        full_name = f"{package}.{module_name}"
        if is_pkg:
            # 子包 → 递归继续扫描
            load_routers(app, full_name, parent_router, is_root=False)
        else:
            # 模块 → 尝试加载 router
            module = importlib.import_module(full_name)
            if hasattr(module, "router") and isinstance(module.router, APIRouter):
                parent_router.include_router(module.router)
    app.include_router(parent_router)