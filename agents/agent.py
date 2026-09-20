from langchain_core.tools import tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich import print
load_dotenv()
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

@tool
def add_numbers(a:int,b:int)->int:
    """Function to add two numbers"""
    return a+b
@tool
def square_number(x:int)->int:
    """Function to find the square of a number"""
    return x*x

print(add_numbers.name)
print(add_numbers.description)
print(add_numbers.args)

tools=[add_numbers,square_number]

agent=create_agent(model=llm,tools=tools,system_prompt="You are an intelligent maths agent")

result=agent.invoke({
    "messages":[
        {
            "role":"human",
            "content":"Square 55 and add 33 with 22 and Find the square of the result"
        }
    ]
})
print(result["messages"][-1].content)