from typing import List

from pydantic import BaseModel


class PackageSearchDTO(BaseModel):
    values: List[str]