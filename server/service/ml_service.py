import asyncio
import time


class MLService:
    def __init__(self):
        pass

    async def search_address(self, address: str):
        loop = asyncio.get_event_loop()

        await loop.run_in_executor(None, time.sleep, 0.1)

        return address + " edited"
