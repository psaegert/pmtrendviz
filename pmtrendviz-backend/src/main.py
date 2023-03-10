import uvicorn
from app.routers import models, predict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def add_routers(app: FastAPI) -> None:
    app.include_router(models.router)
    app.include_router(predict.router)


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    add_routers(app)
    return app


if __name__ == '__main__':
    uvicorn.run("main:create_app", host='0.0.0.0', reload=True, port=8000)
