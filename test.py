import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# prompt = PromptTemplate.from_template("What is LangChain? Explain in one sentence.")
# chain = prompt | llm
# res = chain.invoke({})   
# print(res.content)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    Document(page_content="Python is a programming language."),
    Document(page_content="Java is widely used for enterprise applications."),
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="RAG allows LLMs to retrieve information from external documents.")
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

results = vectorstore.similarity_search(
    "What is RAG?",
    k=2
)

for result in results:
    print("\nResult:")
    print(result.page_content)