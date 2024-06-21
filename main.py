import asyncio
from app.services.similarity_search_agent import SimilaritySearchAgent

async def main():
    ssa = SimilaritySearchAgent()
    relevant_candidates = await ssa.langchain_answer()
    print(relevant_candidates)

if __name__ == "__main__":
    asyncio.run(main())

