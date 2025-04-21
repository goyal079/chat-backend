from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import tempfile
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pinecone

load_dotenv()
# Initialize FastAPI
app = FastAPI()

# Allow CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables / Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Debug prints
print("Pinecone Environment:", PINECONE_ENV)
print("Index Name:", INDEX_NAME)

# Initialize Pinecone with the new client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

# Initialize OpenAI embeddings
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

def get_vectorstore(docs, project_id):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Create vectorstore with the new Pinecone client
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=INDEX_NAME,
        namespace=project_id,
        api_key=PINECONE_API_KEY
    )
    return vectorstore

@app.post("/upload")
async def upload_documents(project_id: str = Form(...), files: List[UploadFile] = File(...)):
    all_docs = []
    print("Project ID:", project_id)
    print("Files:", files)
    for file in files:
        contents = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        docs = load_document(tmp_path)
        all_docs.extend(docs)
        os.remove(tmp_path)

    # Create vectorstore with the new Pinecone client
    vectorstore = PineconeVectorStore.from_documents(
        documents=all_docs,
        index_name=INDEX_NAME,
        embedding=embedding,
        namespace=project_id,
    )
    return {"status": "success", "message": f"Uploaded {len(files)} file(s) to project {project_id}"}

class ChatRequest(BaseModel):
    project_id: str
    message: str
    chat_id: str = None

chat_memory_store = {}

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    retriever = PineconeVectorStore(
        index=pinecone_index,
        embedding=embedding,
        namespace=payload.project_id
    ).as_retriever()

    memory = chat_memory_store.get(payload.chat_id)
    if memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        if payload.chat_id:
            chat_memory_store[payload.chat_id] = memory

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    response = qa_chain.run(payload.message)
    return {"response": response}
