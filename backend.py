# backend.py

from fastapi import FastAPI
from pydantic import BaseModel

# Import the compiled LangGraph
from project import app as workflow


# =========================================================
# LANGGRAPH APP
# =========================================================

# Streamlit imports this:
# from backend import app as workflow

app = workflow


# =========================================================
# FASTAPI APP
# =========================================================

api = FastAPI(
    title="Smart College Assistant API",
    description="FastAPI backend for the Smart College Assistant",
    version="1.0.0"
)


# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):
    message: str
    programme: str = "B.Tech"


# =========================================================
# ROOT
# =========================================================

@api.get("/")
def home():
    return {
        "message": "Smart College Assistant API is running"
    }


# =========================================================
# CHAT
# =========================================================

@api.post("/chat")
def chat(request: ChatRequest):

    result = app.invoke({
        "programme": request.programme,

        "messages": [
            ("user", request.message)
        ],

        "query_type": "",

        "retrieved_context": ""
    })

    response = result["messages"][-1].content

    return {
        "response": response,
        "query_type": result.get("query_type", "general")
    }