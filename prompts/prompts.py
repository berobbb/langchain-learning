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
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

prompt = PromptTemplate.from_template("Explain the topic in one sentence :{topic}")
chain = prompt | llm
res = chain.invoke({"topic":"cricket"})   
# print(res.content)

chat_prompt=ChatPromptTemplate(
    [
        ("system","You are a helpful Financial assistant"),
        ("placeholder","{msgs}")
    ]
)
resp=chat_prompt.invoke({"msgs": [("human", "Hi")]})
# print(resp)

prompt=ChatPromptTemplate.from_messages(
    [("user","Create a small jingle on the topic :{topic}")]
)
chain=prompt|llm|StrOutputParser()
res=chain.invoke({"topic":"Indian Cinema"})
# print(res)

prompt=ChatPromptTemplate.from_messages(
    [("user","Respond to the following question :{qs}")]
)

qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "Who is the first Nobel prize winner?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "where is Taj Mahal Located?"}
]
chain=prompt|llm|StrOutputParser()
res = chain.invoke(qs)
# print(res)

documents = [
    Document(page_content="The stock market saw a significant rise today.", metadata={"source": "CNN", "category": "Finance"}),
    Document(page_content="A new species of frog was discovered in the Amazon.", metadata={"source": "NatGeo", "category": "Science"}),
]

prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")
chain=create_stuff_documents_chain(llm,prompt)
resp=chain.invoke({"context":documents})
print(resp)