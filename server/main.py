import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
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

ml_service = MLService()


@app.get("/check")
def check():
    return "Success"


@app.get("/search")
async def search(query: str):
    return await ml_service.search_address(fix_lang_text_problems(query, "./utils/dict.txt"))


@app.get("/autocomplete")
async def autocomplete(query: str):
    return await ml_service.search_address(fix_lang_text_problems(query, "./utils/dict.txt"))


@app.post("/package_search")
def package_search(data: PackageSearchDTO):
    return data.values


if __name__ == '__main__':
    uvicorn.run(app, port=8000)
