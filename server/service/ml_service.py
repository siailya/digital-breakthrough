import asyncio
import time

from gensim.models import FastText


class MLService:
    def __init__(self, model_path):
        self.model = None
        self.load_fasttext_model(model_path)

    def load_fasttext_model(self, model_path: str):
        self.model = FastText.load(model_path)

    async def search_address(self, address: str):
        loop = asyncio.get_event_loop()

        await loop.run_in_executor(None, time.sleep, 0.1)

        return address + " edited"
