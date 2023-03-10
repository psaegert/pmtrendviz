from fastapi import APIRouter
from pydantic import BaseModel

from pmtrendviz.app.trend import TrendGenerator


class PredictionParameters(BaseModel):
    model_name: str
    query: str | None = None
    resolution: str
    num_clusters: int
    distance: str
    ignore_top_n_dates: int


router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)

tg = TrendGenerator()


@router.post('/')
def get_graph_data(pred_params: PredictionParameters):  # type: ignore

    tg.resolution = pred_params.resolution
    tg.n_closest_clusters = pred_params.num_clusters
    tg.distance = pred_params.distance
    if tg.model_name != pred_params.model_name:
        tg.load_model(pred_params.model_name)
    tg.query = pred_params.query
    tg.ignore_top_n_dates = pred_params.ignore_top_n_dates
    return tg.generate_figure()
