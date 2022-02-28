import imp
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.decomposition import non_negative_factorization

from src.database.connectors import MongoConnector
from src.labeling_manager import LabelingManager

app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="src/frontend/templates")

user = os.getenv("MONGO_INITDB_ROOT_USERNAME")
password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
conn = MongoConnector(
    user,
    password,
    host="157.90.167.200",
    port=27017,
    db="data_labeling",
    collection="dev_coll",
)

manager = LabelingManager(conn, None)


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    item = manager.get_sample()
    return templates.TemplateResponse("index.html", {"request": request, "item": item})
