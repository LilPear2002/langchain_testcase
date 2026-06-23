from fastapi import APIRouter

from app.api import agent, chat, dashboard, document, project

api_router = APIRouter()
api_router.include_router(project.router)
api_router.include_router(document.router)
api_router.include_router(chat.router)
api_router.include_router(agent.router)
api_router.include_router(dashboard.router)
