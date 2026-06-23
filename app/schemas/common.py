from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Msg(BaseModel):
    message: str


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    size: int


class ErrorOut(BaseModel):
    code: int
    message: str
