from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.tools.embedding_tool import EmbeddingTool
from app.tools.reasoning_tool import ReasoningTool

class SimilaritySearchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=settings.OPENAI_API_KEY)
        self.prompt = hub.pull("hwchase17/structured-chat-agent")
        self.tools = [
            EmbeddingTool(),
            ReasoningTool()
        ]

    def langchain_answer(self):
        agent = create_structured_chat_agent(self.llm, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        return agent_executor.invoke({"input": f"What are the best fitting candidates? Give information about them "
                                               f"and reason why they are the best fitting."
                                               f"Path to candidates: 'data/candidates.json', path to pdf: "
                                               f"'data/course_Description.pdf'"})