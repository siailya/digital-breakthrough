import json

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException
from starlette.middleware.cors import CORSMiddleware

from data.dto import PackageSearchDTO
from service.ml_service import MLService
from utils.language_utils import fix_lang_text_problems

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_service = MLService("./model/best_fast_text.model")


@app.get("/api/check")
def check():
    return "Success"


@app.get("/api/search")
async def search(query: str):
    return await ml_service.search_address(fix_lang_text_problems(query, "./utils/dict.txt"))


@app.get("/api/autocomplete")
async def autocomplete(query: str):
    return await ml_service.search_address(fix_lang_text_problems(query, "./utils/dict.txt"))


@app.post("/api/package_search")
def package_search(data: PackageSearchDTO):
    return data.values


@app.post("/api/file_process")
async def file_process(file: UploadFile):
    if file.filename.split(".")[-1] == "txt":
        content = (await file.read()).decode("utf-8").replace("\r", "").split("\n")
    elif file.filename.split(".")[-1] == "json":
        content = json.loads((await file.read()).decode("utf-8"))
    else:
        return HTTPException(415)

    result = []
    for address in content:
        result.append({
            "original_address": address,
            "search_results": await ml_service.search_address(fix_lang_text_problems(address, "./utils/dict.txt"))
        })

    return result


if __name__ == '__main__':
    uvicorn.run(app, port=8000)
