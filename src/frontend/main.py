import imp
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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
    try:
        item = manager.get_sample()
    except RuntimeError:
        print("Done. No more items to label.")
        return RedirectResponse("/done")

    return templates.TemplateResponse("index.html", {"request": request, "item": item})


@app.get("/done", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("done.html", {"request": request})


@app.get("/label/{id}/{label}")
async def clicked(request: Request, id: str, label: str):
    manager.update_one(id_=id, label=label)
    return RedirectResponse("/")
