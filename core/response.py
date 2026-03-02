from pydantic import BaseModel
from typing import Generic, TypeVar, Optional
import numpy as np
T = TypeVar("T")


class R(BaseModel, Generic[T]):
    """
    统一返回格式
    """

    code: int
    msg: str
    data: Optional[T] = None

    @staticmethod
    def success(data: T = None, msg: str = "success"):
        return R(code=200, msg=msg, data=data)

    @staticmethod
    def fail(msg: str = "fail", code: int = -1, data: T = None):
        return R(code=code, msg=msg, data=data)
