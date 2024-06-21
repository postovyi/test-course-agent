from langchain.tools import BaseTool
from app.core.config import settings
import fitz
import json
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


class EmbeddingTool(BaseTool):
    name = "embedding_tool"
    description = ("Use this tool to create embeddings for the texts and candidates descriptions."
                   "You will need them to find similarities between candidates and the text. Do not give any action input."
                   "Do not use the whole text for the context.")

    def __init__(self):
        super().__init__()

    @staticmethod
    def extract_text_from_pdf(pdf_path):
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text

    @staticmethod
    def extract_candidates_data(candidates_path):
        with open(candidates_path, 'r') as file:
            candidates = json.load(file)
        candidates_dict = {}
        for candidate in candidates:
            candidate_description = ""
            for k in candidate:
                param_desc = f"{k.upper()}:{candidate[k]}"
                candidate_description += param_desc
                candidate_description += "\n"
            candidates_dict[candidate["id"]] = candidate_description
        return candidates_dict

    @staticmethod
    def get_relevance_score(embedding1, embedding2):
        return 1 - cosine(embedding1, embedding2)

    async def _arun(self, text):
        pdf_path = "data/course_Description.pdf"
        candidates_path = "data/candidates.json"
        model = SentenceTransformer('all-MiniLM-L6-v2')
        text = self.extract_text_from_pdf(pdf_path)
        candidates = self.extract_candidates_data(candidates_path)
        text_embeddings = model.encode(text)
        candidates_embeddings = {}
        for c in candidates:
            candidates_embeddings[c] = model.encode(candidates[c])
        for k in candidates_embeddings:
            candidates_embeddings[k] = self.get_relevance_score(text_embeddings, candidates_embeddings[k])

        candidates = dict(sorted(candidates_embeddings.items(), key=lambda item: item[1], reverse=True)[:10])
        return candidates
    def _run(self):
        pass