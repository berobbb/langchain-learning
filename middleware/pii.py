from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import PIIMiddleware
from rich import print
load_dotenv()
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
@tool
def check_account_balance(account_id: str) -> str:
	"""Retrieves the current balance for a specific account ID."""
	
	# In actual use case: This would call a secure banking API
	return f"The balance for account {account_id} is $12,450.00."
@tool
def log_support_ticket(priority: str, category: str) -> str:
	
	"""Creates a support ticket in the internal CRM."""
	
	return f"Ticket created: [Priority: {priority}] [Category: {category}]."
tools = [check_account_balance, log_support_ticket]

agent = create_agent(model=llm,
				tools=tools,
				system_prompt=(
				"You are a secure banking assistant. "
				"Sensitive data in your input has been masked for security. "
				"Process the user's request using the available tools."
				),
				middleware=[
				# Strategy 'redact' removes the info entirely
				PIIMiddleware("email", strategy="redact", apply_to_input=True),
				# Strategy 'mask' replaces characters (e.g., --****-1234)
				PIIMiddleware("credit_card", strategy="redact", apply_to_input=True),
				],
			)
def run_secure_session(user_query: str):
	print(f"--- Processing Request ---\nRaw Input: {user_query}\n")
	
	# The agent.invoke handles the middleware logic internally
	# It redacts the PII before the LLM 'sees' the message
	
	response = agent.invoke({"messages": [("user", user_query)]})
	print(response['messages'])
	
	# Extract the final AI message from the conversation history
	final_output = response["messages"][-1].content
	return final_output
raw_input = ( "My email is vip-client@gmail.com."
      "Check my balance for account ACC-990 and log a 'High' priority ticket for billing."
      )
result = run_secure_session(raw_input)
print("--- Agent Final Response ---")
print(result)
