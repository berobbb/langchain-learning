from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")
from langchain_classic.chains import RetrievalQA
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

loader=PyPDFLoader(r"Cloud.pdf")
document=loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        
    chunk_overlap=100,       
    separators=["\n\n", "\n", ".", " ", ""]  
)

docs=text_splitter.split_documents(document)
db=Chroma.from_documents(docs,embeddings)

retriever=db.as_retriever(search_type="similarity",search_kwargs={"k":3})


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
query = "What is cloud computing ?"
result = qa.invoke(query)
print(result['result'])
