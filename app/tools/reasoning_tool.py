from langchain.tools import BaseTool
import json
from typing import List, Dict


class ReasoningTool(BaseTool):
    name = "reasoning_tool"
    description = ("Use this tool to reason why the selected candidates are the best fitting and return detailed "
                   "information about them. You can find information by looking up the candidates with the given IDs."
                   "Mandatory field in input is a list with candidate_ids.")

    def __init__(self):
        super().__init__()

    @staticmethod
    def extract_candidates_data(data: List[str]) -> Dict[str, str]:
        with open("data/candidates.json", 'r') as file:
            candidates = json.load(file)
        candidates_dict = {}
        for candidate in candidates:
            if candidate["id"] in data:
                candidates_dict[candidate["id"]] = candidate["name"] + " " + candidate["summary"]
        return candidates_dict

    async def _arun(self, candidate_ids: List[str]) -> Dict[str, str]:
        return self.extract_candidates_data(candidate_ids)

    def _run(self, candidate_ids: List[str]) -> Dict[str, str]:
        return self.extract_candidates_data(candidate_ids)
