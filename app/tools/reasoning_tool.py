from langchain.tools import BaseTool
from app.core.config import settings
import fitz
import json
from app.schemas.schemas import ReasoningSchema
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


class ReasoningTool(BaseTool):
    name = "embedding_tool"
    description = ("Use this tool to reason why the selected candidates are the best fitting.")

    def __init__(self):
        super().__init__()

    @staticmethod
    def extract_candidates_data(candidates_path, data: ReasoningSchema):
        with open(candidates_path, 'r') as file:
            candidates = json.load(file)
        candidates_dict = {}
        for candidate in candidates:
            if candidate["id"] in data:
                candidates_dict[candidate["id"]] = candidate["name"] + " " + candidate["summary"]
        return candidates_dict

    @staticmethod
    def get_relevance_score(embedding1, embedding2):
        return 1 - cosine(embedding1, embedding2)

    async def _arun(self, data: ReasoningSchema):
        candidates_path = "data/candidates.json"
        candidates = self.extract_candidates_data(candidates_path, data)
        return candidates
    def _run(self):
        pass