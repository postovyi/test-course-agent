from pydantic import BaseModel
from typing import List


class ReasoningSchema(BaseModel):
    best_fits: List[str]
