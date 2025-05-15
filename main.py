import os  # built-in module for interacting with the operating system
import logging  # built-in module for logging messages
from fastapi import FastAPI, HTTPException  # FastAPI framework and HTTP exception class
from pydantic import BaseModel  # Pydantic base class for data validation
from typing import List  # for type annotations

from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware  # for enabling CORS policy

from jose import jwt
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from fastapi import Query

import pinecone  # Pinecone vector database client
from pinecone import Pinecone as PineconeClient  # alias for Pinecone client class

# LangChain imports for embeddings and LLMs
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # embeddings and chat wrapper
from langchain_pinecone import Pinecone as PineconeVectorStore  # Pinecone vectorstore wrapper
from langchain.chains import RetrievalQA  # RAG QA chain
from langchain.prompts import (  # prompt template classes
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# JWT Settings
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 #minutes

# Load variables from .env file
load_dotenv()

# Retrieve variables
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
NOVITA_API_KEY   = os.getenv("NOVITA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")
MONGO_URI        = os.getenv("MONGO_URI")

# MongoDB connection
client = MongoClient(MONGO_URI)

db = client["LegalApp"]
users_collection = db["users"]
chat_collection = db["chat_history"]

# Pydantic models
class LoginModel(BaseModel):
  email: str
  password: str

class TokenResponse(BaseModel):
  access_token: str
  token_type: str
  user: dict

# Request schema for query endpoint\
class QueryRequest(BaseModel):
  query: str  # user question as string
  id_user: int  # user ID for tracking
    
# Response schema for QA endpoint
class QAResponse(BaseModel):
  question: str  # original question
  answer: str  # generated answer
  #sources: List[str]  # list of source doc references
    
class SignupModel(BaseModel):
  firstName: str
  lastName: str
  email: str
  password: str
  confirmPassword: str
  state: str
  isLegalProfessional: bool
    
class HistoryItem(BaseModel):
  question: str
  answer: str
  timestamp: datetime

app = FastAPI()

# — Add CORS middleware —
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:4200"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Testing the database connection endpoint
@app.get("/test-connection")
def test_connection():
  try:
    # Ping the server
    client.admin.command("ping")
    collections = db.list_collection_names()
    return {
      "status": "success",
      "message": "Connected to MongoDB",
      "collections": collections
    }
  except ConnectionFailure as e:
    return {
      "status": "error",
      "message": f"Connection failed: {str(e)}"
    }

# Routes
@app.post("/login", response_model=TokenResponse)
def login(login_data: LoginModel):
  # print(f"Received login request: {login_data.email}")
  user = users_collection.find_one({"email": login_data.email})

  if not user or user["password"] != login_data.password:
    raise HTTPException(status_code=401, detail="Invalid email or password")

  token_data = {
    "sub": user["email"],
    "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
  }
  token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

  # Prepare user data excluding password
  user.pop("password")
  user["_id"] = str(user["_id"])  # Convert ObjectId to string

  return {
    "access_token": token,
    "token_type": "bearer",
    "user": user
  }

@app.post("/signup", response_model=TokenResponse)
def signup(user_data: SignupModel):
  if user_data.password != user_data.confirmPassword:
    raise HTTPException(status_code=400, detail="Passwords do not match")

  # Check if email already exists
  existing_user = users_collection.find_one({"email": user_data.email})
  if existing_user:
    raise HTTPException(status_code=400, detail="Email already registered. Try signing in or use another email.")

  user = {
    "firstName": user_data.firstName,
    "lastName": user_data.lastName,
    "email": user_data.email,
    "password": user_data.password,  # hash this
    "state": user_data.state,
    "isLegalProfessional": user_data.isLegalProfessional,
    "createdAt": datetime.utcnow()
  }

  result = users_collection.insert_one(user)

  # Issue JWT token
  token_data = {
    "sub": user["email"],
    "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
  }
  token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

  user["_id"] = str(result.inserted_id)
  user.pop("password")  # do not send password back

  return {
    "access_token": token,
    "token_type": "bearer",
    "user": user
  }

# — Logging —
logging.basicConfig(level=logging.INFO)  # configure root logger to INFO level



# Ensure all required credentials are present
if not all([OPENAI_API_KEY, NOVITA_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
    raise RuntimeError("Missing one or more required API keys/env vars")  # terminate if missing

# — 2. Initialize Pinecone & patch for LangChain —
pine_client = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)  # instantiate Pinecone client
index = pine_client.Index("mohsin")  # connect to existing Pinecone index named 'mohsin'
pinecone.Index = index.__class__  # monkey-patch Pinecone SDK so LangChain sees correct Index class

# — 3. Embeddings model —
embed_model = OpenAIEmbeddings(
    model="text-embedding-3-large",  # specify embedding model
    openai_api_key=OPENAI_API_KEY  # pass API key to embeddings wrapper
)

# — 4. VectorStore wrapper —
vectorstore = PineconeVectorStore(
    index=index,  # Pinecone index instance
    embedding=embed_model,  # embedding model to use
    text_key="page_content",  # key in metadata for text content
    namespace="legal-docs"  # logical namespace within Pinecone index
)

# — 5. Retriever —
# Convert vectorstore to retriever for RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # return top 3 nearest neighbors

# — 6. LLM via Novita.ai —
llm = ChatOpenAI(
  model_name="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",  # specify LLM
  openai_api_key=NOVITA_API_KEY,  # use Novita API key
  openai_api_base="https://api.novita.ai/v3/openai",  # endpoint for Novita API
  temperature=0.2,  # lower randomness for deterministic answers
  max_tokens=256,  # limit of generated tokens
)

# — 7. Prompt templates for prompt engineering —
# Build a chain-of-thought style prompt for legal context
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are expert lawyer act like know everything with a focus on legal expertise specialized in U.S state corporate law or guide the help like a lawyer."
    ),
    AIMessagePromptTemplate.from_template(
        "Guide the user to the best of your ability and provide guidenace on the law."
        "If the query is not really related to law, say 'I am not a lawyer, but I can help you with your query.'"
    ),
    HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n \nUser: {question}\nAssistant:"  # user input placeholder
    ),
])

# — 8. Build RAG QA chain with custom prompt —
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # LLM instance
    chain_type="stuff",  # 'stuff' method for combining context
    retriever=retriever,  # vector retriever
    return_source_documents=True,  # return docs for source attribution
    chain_type_kwargs={"prompt": prompt}  # custom prompt
)

@app.post("/qa", response_model=QAResponse)
async def qa_endpoint(req: QueryRequest):
    logging.info(f"Incoming query: {req.query}")  # log received query
    try:
        # a) Embed the query
        q_emb = embed_model.embed_query(req.query)  # generate query embedding
        # b) Get RAG‑based answer
        res = qa_chain.invoke({"query": req.query})  # run QA chain
        
        conversation_id = str(ObjectId())  # Generate unique conversation ID
                
        # Store chat history in MongoDB
        await chat_collection.insert_one({
          "user_id": str(req.id_user),
          "question": req.query,
          "answer": res["result"],
          "timestamp": datetime.utcnow(),
          "conversation_id": conversation_id
        })
        
        # c) Collect sources
        #sources = [
            #f"{doc.metadata.get('source','?')} (chunk {doc.metadata.get('chunk_id','?')})"
            #for doc in res["source_documents"]  # iterate returned docs
        #]
        return QAResponse(
            question=req.query,
            answer=res["result"],
            #sources=sources
        )  # return structured response
    except Exception as e:
        logging.error("Error in /qa", exc_info=True)  # log exception details
        raise HTTPException(status_code=500, detail=str(e))  # return HTTP 500 with error message

from bson import ObjectId
# API to get user's chat history    
@app.get("/history", response_model=List[HistoryItem])
async def get_history(user_id: str = Query(...)):
    docs = list(chat_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(100))
    return [{"question": d["question"], "answer": d["answer"], "timestamp": d["timestamp"], "conversation_id": d["conversation_id"]} for d in docs]


# Deleting a user account
@app.delete("/delete-user/{user_id}")
async def delete_user(user_id: str):
    try:
        # Validate and convert user_id to ObjectId
        if not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=400, detail="Invalid user ID format.")

        # Try to delete the user from MongoDB
        result = users_collection.delete_one({"_id": ObjectId(user_id)})

        if result.deleted_count == 0:
            # User with given ID not found
            raise HTTPException(status_code=404, detail="User not found.")

        # Success response
        return {"message": f"User with ID {user_id} deleted successfully."}

    except HTTPException as e:
        # Re-raise known HTTP exceptions
        raise e

    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")