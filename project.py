import os
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal

from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0.4
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# ============================================================
# STEP 1 - BUILDING RAG RETRIEVERS
# ============================================================

def build_retriever(pdf_path: str):
    """
    Loads a PDF, splits its content into smaller chunks,
    creates embeddings for those chunks, stores them in FAISS,
    and returns a retriever.
    """

    # Convert to Path for reliable cross-platform handling
    pdf_file = Path(pdf_path)

    # Check that the PDF actually exists
    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_file}"
        )

    if not pdf_file.is_file():
        raise FileNotFoundError(
            f"Path is not a file: {pdf_file}"
        )

    # Load PDF
    loader = PyPDFLoader(str(pdf_file))

    document = loader.load()

    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(document)

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Return retriever
    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


# ============================================================
# PDF DIRECTORY
# ============================================================

# This gets the directory where project.py is located.
#
# Local:
# C:\...\smart-college-assistant\
#
# Streamlit Cloud:
# /mount/src/smart-college-assistant/
#
# Therefore this works on both environments.

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "PDFs"


# ============================================================
# PDF FILE PATHS
# ============================================================

ACADEMICS_PDF = PDF_DIR / "01_BKBIET_Academics.pdf"

FEES_PDF = PDF_DIR / "02_BKBIET_Fees_Hostel_Scholarships.pdf"

PLACEMENTS_PDF = PDF_DIR / "03_BKBIET_Placements_Recruiters.pdf"

FACULTY_PDF = PDF_DIR / "04_BKBIET_Faculty_Staff_Leadership.pdf"


# ============================================================
# BUILD RETRIEVERS
# ============================================================

academic_retriever = build_retriever(
    str(ACADEMICS_PDF)
)

fees_retriever = build_retriever(
    str(FEES_PDF)
)

placement_retriever = build_retriever(
    str(PLACEMENTS_PDF)
)

faculty_retriever = build_retriever(
    str(FACULTY_PDF)
)


# ============================================================
# STEP 2 - STATE
# ============================================================

class State(TypedDict):

    programme: str

    messages: Annotated[list, add_messages]

    query_type: str

    retrieved_context: str


# ============================================================
# STEP 3 - CLASSIFIER NODE
# ============================================================

def classifier_node(state: State) -> dict:
    """
    Looks at the latest user message and classifies the query
    into academic, fees, placement, faculty, or general.
    """

    last_message = state["messages"][-1].content

    prompt = f"""
You are a query classifier for a Smart College Chatbot.

Analyze the user's latest query and classify it into exactly
one of these categories:

- academic — courses, programs, departments, curriculum,
  subjects, admission-related academic information,
  academic rules, or college academics.

- fees — tuition fees, hostel fees, scholarships,
  financial information, hostel facilities, or payment-related queries.

- placement — placements, recruiters, companies, packages,
  placement statistics, internships, or career-related college information.

- faculty — faculty members, teachers, staff, leadership,
  departments' faculty, or college administration.

- general — greetings, casual conversation, general questions,
  general AI interaction, or queries that do not belong
  to the other categories.

Return only the category name in lowercase.

Possible outputs:

academic
fees
placement
faculty
general

Do not provide an explanation or any additional text.

User query:
{last_message}
"""

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"

    elif "fees" in category:
        category = "fees"

    elif "placement" in category:
        category = "placement"

    elif "faculty" in category:
        category = "faculty"

    else:
        category = "general"

    return {
        "query_type": category
    }


# ============================================================
# ACADEMIC NODE
# ============================================================

def academic_node(state: State) -> dict:
    """
    Retrieves relevant academic information from the academic
    knowledge base using the user's query.
    """

    query = state["messages"][-1].content

    documents = academic_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# FEES NODE
# ============================================================

def fees_node(state: State) -> dict:
    """
    Retrieves relevant information about fees, hostel,
    and scholarships from the fees knowledge base.
    """

    query = state["messages"][-1].content

    documents = fees_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# PLACEMENT NODE
# ============================================================

def placement_node(state: State) -> dict:
    """
    Retrieves relevant placement and recruiter information
    from the placement knowledge base using the user's query.
    """

    query = state["messages"][-1].content

    documents = placement_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# FACULTY NODE
# ============================================================

def faculty_node(state: State) -> dict:
    """
    Retrieves relevant information about faculty, staff,
    and college leadership from the faculty knowledge base.
    """

    query = state["messages"][-1].content

    documents = faculty_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# GENERAL NODE
# ============================================================

def general_node(state: State) -> dict:
    """
    Handles general conversations and queries that do not
    require retrieval from the college knowledge bases.
    """

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# ============================================================
# RESPONSE NODE
# ============================================================

def response_node(state: State) -> dict:
    """
    Generates the final answer using the LLM.

    If retrieval is not required, the LLM answers using
    general knowledge. Otherwise, the LLM uses the retrieved
    context from the relevant college knowledge base.
    """

    query = state["messages"][-1].content

    programme = state.get(
        "programme",
        "Unknown"
    )

    context = state.get(
        "retrieved_context",
        ""
    )

    # --------------------------------------------------------
    # GENERAL RESPONSE
    # --------------------------------------------------------

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = (
            f"You are a friendly Smart College Assistant "
            f"helping a {programme} student. "
            f"Answer the following question using your general "
            f"knowledge. Keep the response clear, helpful, "
            f"and conversational.\n\n"
            f"Question: {query}"
        )

    # --------------------------------------------------------
    # RAG RESPONSE
    # --------------------------------------------------------

    else:

        prompt = (
            f"You are a Smart College Assistant helping a "
            f"{programme} student. "

            f"Use the following context from the college "
            f"knowledge base to answer the question accurately. "

            f"Do not make up information that is not supported "
            f"by the context. "

            f"If the context does not contain enough information "
            f"to answer the question, clearly say that the "
            f"available information is insufficient.\n\n"

            f"Context:\n{context}\n\n"

            f"Question:\n{query}\n\n"

            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)

    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }


# ============================================================
# STEP 4 - ROUTER FUNCTION
# ============================================================

def route_query(
    state: State
) -> Literal[
    "academic_rag",
    "fees_rag",
    "placement_rag",
    "faculty_rag",
    "general"
]:

    query_type = state["query_type"]

    if query_type == "academic":
        return "academic_rag"

    elif query_type == "fees":
        return "fees_rag"

    elif query_type == "placement":
        return "placement_rag"

    elif query_type == "faculty":
        return "faculty_rag"

    else:
        return "general"


# ============================================================
# STEP 5 - BUILDING LANGGRAPH
# ============================================================

graph = StateGraph(State)


# ============================================================
# ADD NODES
# ============================================================

graph.add_node(
    "classifier",
    classifier_node
)

graph.add_node(
    "academic_rag",
    academic_node
)

graph.add_node(
    "fees_rag",
    fees_node
)

graph.add_node(
    "placement_rag",
    placement_node
)

graph.add_node(
    "faculty_rag",
    faculty_node
)

graph.add_node(
    "general",
    general_node
)

graph.add_node(
    "response",
    response_node
)


# ============================================================
# EDGES
# ============================================================

graph.add_edge(
    START,
    "classifier"
)


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

graph.add_conditional_edges(
    "classifier",
    route_query,
    {
        "academic_rag": "academic_rag",
        "fees_rag": "fees_rag",
        "placement_rag": "placement_rag",
        "faculty_rag": "faculty_rag",
        "general": "general",
    }
)


# ============================================================
# RAG NODES -> RESPONSE
# ============================================================

graph.add_edge(
    "academic_rag",
    "response"
)

graph.add_edge(
    "fees_rag",
    "response"
)

graph.add_edge(
    "placement_rag",
    "response"
)

graph.add_edge(
    "faculty_rag",
    "response"
)

graph.add_edge(
    "general",
    "response"
)


# ============================================================
# RESPONSE -> END
# ============================================================

graph.add_edge(
    "response",
    END
)


# ============================================================
# COMPILE GRAPH
# ============================================================

app = graph.compile()