"""
Smart College Assistant — Streamlit UI

Drop this in as app.py and run:
streamlit run app.py

This file only handles the FRONTEND. It expects your existing LangGraph
backend (the classifier -> router -> RAG nodes -> response graph) to be
importable as a compiled graph called `app` (the same name your script
used: `app = graph.compile()`).

>>> Save your backend script as `backend.py` in the same folder as this
>>> file (or change the import line below to match whatever you name it).
"""

import streamlit as st

# ---------------------------------------------------------------------
# Import your LangGraph backend.
# Adjust this to match your actual filename, e.g.:
#   from graph import app as workflow
#   from smart_college_graph import app as workflow
from backend import app as workflow

# =======================================================================
# PAGE CONFIG
# =======================================================================

st.set_page_config(
    page_title="Smart College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =======================================================================
# THEME / CSS  (dark, student-friendly, one accent color)
# =======================================================================

st.markdown("""
<style>
    :root{
        --bg: #0F1117;
        --panel: #161A22;
        --panel-raised: #1C202B;
        --border: #262B38;
        --text-secondary: #8B93A6;
        --accent: #7C87F5;
    }

    .stApp{
        background-color: var(--bg);
    }

    section[data-testid="stSidebar"]{
        background-color: var(--panel);
        border-right: 1px solid var(--border);
    }

    /* chat bubbles */
    div[data-testid="stChatMessage"]{
        background-color: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 4px 6px;
    }

    /* suggestion / category buttons */
    div.stButton > button{
        background-color: var(--panel-raised);
        color: #E7E9EE;
        border: 1px solid var(--border);
        border-radius: 8px;
        text-align: left;
        width: 100%;
    }

    div.stButton > button:hover{
        border-color: var(--accent);
        color: var(--accent);
    }

    .status-pill{
        display:inline-flex;
        align-items:center;
        gap:6px;
        font-size:12px;
        color: var(--text-secondary);
        background: var(--panel-raised);
        border: 1px solid var(--border);
        padding: 3px 10px;
        border-radius: 999px;
    }

    .status-pill .dot{
        width:6px;
        height:6px;
        border-radius:50%;
        background:#7FCB8F;
    }

    .route-tag{
        font-size: 11px;
        color: var(--text-secondary);
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)


# =======================================================================
# SESSION STATE
# =======================================================================

# `conversations` holds every chat the user has started this session, so
# the sidebar can list history and let them switch back to an old one.

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_id" not in st.session_state:
    st.session_state.current_id = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "last_error_prompt" not in st.session_state:
    st.session_state.last_error_prompt = None


def new_chat():
    import uuid

    cid = str(uuid.uuid4())

    st.session_state.conversations[cid] = {
        "title": None,
        "messages": []
    }

    st.session_state.current_id = cid


def current_messages():
    if st.session_state.current_id is None:
        return None

    return st.session_state.conversations[
        st.session_state.current_id
    ]["messages"]


if st.session_state.current_id is None:
    new_chat()


# =======================================================================
# SIDEBAR
# =======================================================================

with st.sidebar:

    st.markdown("### 🎓 Smart College Assistant")

    st.caption("BKBIET · AI helper")

    if st.button("➕  New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown("**Browse by topic**")

    category_prompts = {
        "📚 Academics":
            "What courses and specializations are available?",

        "💰 Fees & Hostel":
            "What are the hostel fees and scholarship options?",

        "💼 Placements":
            "Which companies recruit from the college?",

        "👨‍🏫 Faculty":
            "Who are the faculty members in the AI department?",
    }

    for label, prompt in category_prompts.items():

        if st.button(
            label,
            key=f"cat_{label}",
            use_container_width=True
        ):
            st.session_state.pending_prompt = prompt
            st.rerun()

    st.markdown("**Recent chats**")

    convos = st.session_state.conversations

    if not any(c["title"] for c in convos.values()):

        st.caption(
            "Your conversations will appear here once you start chatting."
        )

    else:

        for cid, convo in reversed(list(convos.items())):

            if convo["title"] is None:
                continue

            is_current = cid == st.session_state.current_id

            if st.button(
                ("▶ " if is_current else "") + convo["title"],
                key=f"hist_{cid}",
                use_container_width=True,
            ):
                st.session_state.current_id = cid
                st.rerun()

    st.markdown("---")

    st.button(
        "⚙️  Settings",
        use_container_width=True,
        disabled=True
    )


# =======================================================================
# TOP BAR
# =======================================================================

top_left, top_right = st.columns([3, 1])

with top_left:

    st.markdown(
        "#### Smart College Assistant "
        "<span class='status-pill'>"
        "<span class='dot'></span> AI Assistant"
        "</span>",
        unsafe_allow_html=True,
    )

with top_right:

    st.markdown(
        "<div style='text-align:right; color:#8B93A6; "
        "font-size:12px; padding-top:14px;'>"
        "B.Tech student"
        "</div>",
        unsafe_allow_html=True,
    )


messages = current_messages()


# =======================================================================
# WELCOME SCREEN
# =======================================================================

if not messages:

    st.markdown(
        "## Hi! 👋 How can I help you today?"
    )

    st.caption(
        "Ask me about academics, fees, hostel, scholarships, placements, "
        "faculty, or anything related to college."
    )

    c1, c2 = st.columns(2)

    suggestions = [

        (
            "📚 Academic Programs",
            "What courses are available?"
        ),

        (
            "💰 Fees & Scholarships",
            "What are the hostel fees?"
        ),

        (
            "💼 Placement Information",
            "Which companies recruit from the college?"
        ),

        (
            "👨‍🏫 Faculty Information",
            "Who are the faculty members in the AI department?"
        ),

    ]

    cols = [c1, c2, c1, c2]

    for col, (label, prompt) in zip(cols, suggestions):

        with col:

            if st.button(
                label,
                key=f"sugg_{label}",
                use_container_width=True
            ):
                st.session_state.pending_prompt = prompt
                st.rerun()


# =======================================================================
# RENDER EXISTING MESSAGES
# =======================================================================

else:

    for msg in messages:

        role = (
            "user"
            if msg["role"] == "user"
            else "assistant"
        )

        avatar = (
            "🧑‍🎓"
            if role == "user"
            else "🎓"
        )

        with st.chat_message(
            role,
            avatar=avatar
        ):

            st.markdown(msg["content"])

            if msg.get("route"):

                st.markdown(
                    f"<div class='route-tag'>{msg['route']}</div>",
                    unsafe_allow_html=True
                )


# =======================================================================
# BACKEND CALL
# =======================================================================

CATEGORY_ROUTE_LABEL = {

    "academic":
        "Academic RAG · Academics knowledge base",

    "fees":
        "Fees RAG · Fees, Hostel & Scholarships knowledge base",

    "placement":
        "Placement RAG · Placements & Recruiters knowledge base",

    "faculty":
        "Faculty RAG · Faculty, Staff & Leadership knowledge base",
}


def run_backend(
    user_text: str,
    history_messages: list
):

    """
    Calls your compiled LangGraph app the same way your CLI loop did.
    `history_messages` is the running ("user"/"ai", text) tuple list your
    State expects.
    """

    lc_history = [
        (
            m["role"]
            if m["role"] == "user"
            else "ai",
            m["content"]
        )
        for m in history_messages
    ]

    lc_history.append(
        ("user", user_text)
    )

    result = workflow.invoke({

        "programme": "B.Tech",

        "messages": lc_history,

        "query_type": "",

        "retrieved_context": "",

    })

    reply_text = result["messages"][-1].content

    query_type = result.get(
        "query_type",
        "general"
    )

    route_label = CATEGORY_ROUTE_LABEL.get(
        query_type
    )

    return reply_text, route_label


def handle_user_message(
    user_text: str
):

    convo = st.session_state.conversations[
        st.session_state.current_id
    ]

    lower = user_text.strip().lower()

    if lower == "clear":

        convo["messages"] = []

        convo["title"] = None

        return

    if lower in (
        "exit",
        "quit",
        "bye"
    ):

        convo["messages"].append({
            "role": "user",
            "content": user_text
        })

        convo["messages"].append({
            "role": "assistant",
            "content":
                "Session ended. Goodbye! 👋 "
                "Start a new chat anytime."
        })

        if convo["title"] is None:

            convo["title"] = (
                user_text[:34]
                + (
                    "…"
                    if len(user_text) > 34
                    else ""
                )
            )

        return

    convo["messages"].append({
        "role": "user",
        "content": user_text
    })

    if convo["title"] is None:

        convo["title"] = (
            user_text[:34]
            + (
                "…"
                if len(user_text) > 34
                else ""
            )
        )

    with st.chat_message(
        "user",
        avatar="🧑‍🎓"
    ):

        st.markdown(user_text)

    with st.chat_message(
        "assistant",
        avatar="🎓"
    ):

        placeholder = st.empty()

        placeholder.markdown(
            "_Thinking..._"
        )

        try:

            reply_text, route_label = run_backend(
                user_text,
                convo["messages"][:-1]
            )

            placeholder.markdown(
                reply_text
            )

            if route_label:

                st.markdown(
                    f"<div class='route-tag'>{route_label}</div>",
                    unsafe_allow_html=True
                )

            convo["messages"].append({

                "role": "assistant",

                "content": reply_text,

                "route": route_label

            })

            st.session_state.last_error_prompt = None

        except Exception as e:

            placeholder.empty()

            st.error(
                "Something went wrong. Please try again."
            )

            st.caption(
                f"Details: {e}"
            )

            st.session_state.last_error_prompt = user_text

            if st.button(
                "Retry",
                key=f"retry_{len(convo['messages'])}"
            ):

                st.session_state.pending_prompt = (
                    st.session_state.last_error_prompt
                )

                st.rerun()


# =======================================================================
# HANDLE PENDING PROMPT
# =======================================================================

if st.session_state.pending_prompt:

    prompt = st.session_state.pending_prompt

    st.session_state.pending_prompt = None

    handle_user_message(prompt)

    st.rerun()


# =======================================================================
# CHAT INPUT
# =======================================================================

user_input = st.chat_input(
    "Ask anything about your college..."
)

if user_input:

    handle_user_message(
        user_input
    )

    st.rerun()


# =======================================================================
# CREATED BY
# =======================================================================

st.markdown(
    "<div style='text-align:center; "
    "color:#8B93A6; "
    "font-size:13px; "
    "margin-top:20px;'>"
    "Created by Dharmesh Sharma"
    "</div>",
    unsafe_allow_html=True
)