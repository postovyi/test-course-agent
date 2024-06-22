from app.services.similarity_search_agent import SimilaritySearchAgent


ssa = SimilaritySearchAgent()
print(ssa.langchain_answer()["output"])
