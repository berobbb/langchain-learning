import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from rich import print
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
store={}
# Called by RunnableWithMessageHistory on every invoke().
# Returns the existing history for a session, or creates an empty one the first time.
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Prompt with three parts:
#  - system message: sets the assistant's behavior
#  - placeholder: past messages get inserted here (name must match history_messages_key)
#  - human message: the current user question (name must match input_messages_key)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{history}"),
    ("human", "{input}"),
])

# Basic chain: prompt is filled, then sent to the LLM
runnable = prompt | llm

# Wrap the chain with memory.
# Before each call: loads the session's history and injects it as "history".
# After each call: saves the new user message and AI reply to that history.
chain = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",       # key in the input dict holding the user's message
    history_messages_key="history",   # matches "{history}" in the prompt
)

# The session_id tells the wrapper which history to load and save.
# Use the same ID to continue a conversation, a new ID to start a fresh one.
config = {"configurable": {"session_id": "1"}}

response = chain.invoke({"input": "my name is sd"}, config=config)

# response is an AIMessage; .content holds just the text
print(response.content)
response = chain.invoke({"input": "what is my name"}, config=config)
print(response.content)