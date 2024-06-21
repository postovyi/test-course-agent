from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_openai import OpenAI
from app.core.config import settings
from app.tools.embedding_tool import EmbeddingTool
from app.tools.reasoning_tool import ReasoningTool

class SimilaritySearchAgent:
    def __init__(self):
        self.llm = OpenAI(openai_api_key=settings.OPENAI_API_KEY)
        self.prompt = hub.pull("hwchase17/react")
        self.tools = [
            EmbeddingTool(),
        ]

    async def langchain_answer(self):
        agent = create_react_agent(self.llm, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        await agent_executor.ainvoke({"input": f"What are the best fitting candidates? Why?"})

