from logging import getLogger

from fastapi import APIRouter

from pmtrendviz.models.args import ModelBaseArgs, PreProcessingArgs
from pmtrendviz.train.args import GeneralArgs

logger = getLogger('pmtrendviz.api.routers.models')

router = APIRouter(
    prefix="/train",
    tags=["train"]
)


@router.get("/args")
def get_train_args() -> dict:  # type: ignore
    return GeneralArgs.schema()


@router.post("/new")
async def new_training(general_args: GeneralArgs, preprocess_args: PreProcessingArgs, model_args: ModelBaseArgs) -> int:  # type: ignore
    print(general_args)
    print(preprocess_args)
    print(model_args)
    return 0
