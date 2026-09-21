from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
search = DuckDuckGoSearchRun()
tools=[search]
memory=InMemorySaver()
agent = create_agent(llm,tools=tools,checkpointer=memory, system_prompt="you are a helpful assistant")
config = {"configurable": {"thread_id": "Mar26"}}
from rich import print
content=agent.invoke( {"messages": [HumanMessage(content="Hi I'm Ram and I live in Dehradun")]}, config)
print(content["messages"][-1].content)
content=agent.invoke( {"messages": [HumanMessage(content="Latest news about where I live")]}, config)
print(content["messages"][-1].content)
