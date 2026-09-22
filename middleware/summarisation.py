from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
@tool
def get_course(course_name:str):
    """Returns the course name asked by the user"""
    courses={
        "DSA":"Love babbar",
        "System Design":"Striver",
        "Web dev":"Chai with code"
    }
    course=courses.get(course_name,"Course currently not availabe")
    return course
@tool
def calculate_installment_plan(total_fee: float, months: int):
	"""Calculates the monthly payment for a student fee plan."""
	if months <= 0:
		return "Duration must be at least 1 month."
	monthly = total_fee / months
	return f"The installment plan is {months} payments of ${monthly:.2f} per month."

tools=[get_course,calculate_installment_plan]

memory=InMemorySaver()
SYSTEM_PROMPT = """You are the professional receptionist for 'FutureTech Institute'.
Your goal is to help prospective students with course info and fee queries.
Be polite, helpful, and concise."""
agent=create_agent(
	model=llm,
	tools=tools,
	checkpointer=memory,
	system_prompt=SYSTEM_PROMPT,
	middleware=[
		SummarizationMiddleware
        (
			model=llm,
			trigger=("tokens",400),# Summarize when context hits 400 tokens
			keep=("messages",60)# Always keep the last 6 messages raw
        ),
    ],
)
from rich import print
config = {"configurable": {"thread_id": "student_001"}}
while True:
	query=input("User_Query:")
	if query=='exit':
		break
	response = agent.invoke(
		{"messages": [{"role": "user", "content":query }]},
		config=config
		)
		
	print(response["messages"][-1].content)
	print(response)
