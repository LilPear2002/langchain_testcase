from app.db.base import Base
from app.db.session import engine
from app.db import models  # noqa: F401  确保模型注册到 Base.metadata


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
