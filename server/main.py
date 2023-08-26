import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from sqlmodel import create_engine

from data.create_database import create_db_if_not_exists
from service.ml_service import MLService
from utils.language_utils import fix_lang_text_problems

load_dotenv()

DATABASE_FILE = os.getenv('SQLITE_DB_PATH') or os.path.abspath(r'database/database.sqlite')
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

app = FastAPI()
ml_service = MLService()

create_db_if_not_exists(DATABASE_FILE)
engine = create_engine(DATABASE_URL, echo=False)


@app.get("/search/")
async def search(query: str):
    return await ml_service.search_address(fix_lang_text_problems(query, "./utils/dict.txt"))


@app.get("/autocomplete/")
async def autocomplete(query: str):
    return await ml_service.search_address(fix_lang_text_problems(query, "./utils/dict.txt"))


if __name__ == '__main__':
    uvicorn.run(app, port=8000)
