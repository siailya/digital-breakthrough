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
    result = await ml_service.search_addresses(fix_lang_text_problems(query, "./utils/dict.txt"))
    return {
        "success": len(result) != 0,
        "query": {
            "address": query
        },
        "result": result
    }


@app.get("/api/autocomplete")
async def autocomplete(query: str):
    result = await ml_service.search_addresses(fix_lang_text_problems(query, "./utils/dict.txt"))
    return list(map(lambda x: x["full_address"], result))


@app.post("/api/package_search")
async def package_search(data: PackageSearchDTO):
    result = []
    for address in filter(lambda x: x != '', data.values):
        result.append({"query": address, "result": await search(address)})

    return result


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
            "query": address,
            "result": await search(address)
        })

    return result


if __name__ == '__main__':
    uvicorn.run(app, port=8000)
