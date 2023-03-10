import argparse

import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import models, predict


def api(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(
        title="PubMed Research Trends",
        description="PubMed Research Trends is a web application that allows users to explore trends in biomedical research. Users can search for topics and visualize the trends in the number of publications over time. The application also provides a topic clustering feature that allows users to explore the topics that are related to a given topic.",
        version="0.1.0",
    )

    app.add_middleware(CORSMiddleware,
                       allow_origins=["*"],
                       allow_credentials=True,
                       allow_methods=["*"],
                       allow_headers=["*"])

    root_router = APIRouter(
        tags=["root"]
    )
    app.include_router(root_router)
    app.include_router(models.router)
    app.include_router(predict.router)

    @root_router.get('/', status_code=200)
    def root() -> str:
        return "Hello World!"

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level='debug')
