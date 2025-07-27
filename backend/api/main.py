
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()


# LangSmith environment setup
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "HealthChatbot"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


# System prompt for the assistant
SYSTEM_PROMPT = """
You are a helpful, knowledgeable assistant specialized in health, diet, and wellness. Use the provided knowledge base to answer user questions accurately and concisely. If a question is about BMI, use the BMI tool. If you don't know the answer, say so honestly.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Load FAISS index and embedding model
EMBEDDING_MODEL = "nomic-embed-text"
INDEX_PATH = os.getenv("INDEX_PATH", "../../faiss_index_pdf")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
db = FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1, "k": 4})


# from langchain.tools.retriever import create_retriever_tool

# retriever_tool = create_retriever_tool(
#     retriever,
#     "retrieve_blog_posts",
#     "Search and return information about Diet and health.",
# )


# --- Tool: Vector Search ---
def vector_search_tool(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if docs:
        return "\n\n".join([d.page_content for d in docs])
    return "No relevant information found in the knowledge base."

# --- Tool: BMI Calculator ---
def calculate_bmi(height_cm: float, weight_kg: float):
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": bmi, "category": category}

def bmi_tool(height_cm: float, weight_kg: float) -> str:
    result = calculate_bmi(height_cm, weight_kg)
    bmi = result['bmi']
    category = result['category']
    if category == "Underweight":
        advice = "This means you are under the recommended weight for your height. Consider consulting a healthcare provider for advice on healthy weight gain."
    elif category == "Normal weight":
        advice = "This means you are within the healthy weight range for your height. Keep up your healthy habits!"
    elif category == "Overweight":
        advice = "This means you are above the recommended weight for your height. Consider a balanced diet and regular exercise."
    else:
        advice = "This means you are in the obese range. It's recommended to consult a healthcare provider for personalized advice."
    return f"The user's BMI is {bmi:.2f}, which means you are in the '{category}' category. {advice}"


# Arxiv Tool setup
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arxiv papers")

# Wikipedia tool setup
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, description="Wikipedia")

# --- LangGraph LLM and Tools Setup ---
from langchain.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Contains relevant information related to health and diet."""
    return vector_search_tool(query)

@tool
def compute_bmi(height_cm: float, weight_kg: float) -> str:
    """Calculates BMI and returns the value and category."""
    return bmi_tool(height_cm, weight_kg)

@tool
def arxiv_search(query: str) -> str:
    """Relevant research papers are available to be fetched."""
    return api_wrapper_arxiv.run(query)

@tool
def wiki_search(query: str) -> str:
    """Searches Wikipedia for relevant articles."""
    return api_wrapper_wiki.run(query)

tools = [search_knowledge_base, compute_bmi, arxiv_search, wiki_search]

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
# qwen-qwq-32b
llm_with_tools = llm.bind_tools(tools=tools)

def tool_calling_llm(state):
    # Pass system prompt explicitly if present in state
    messages = state["messages"]
    system_prompt = state.get("system_prompt")
    if system_prompt:
        # Ensure system prompt is the first message
        if not (messages and isinstance(messages[0], SystemMessage) and messages[0].content == system_prompt):
            messages = [SystemMessage(content=system_prompt)] + messages
    return {"messages": [llm_with_tools.invoke(messages)]}

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    system_prompt: Optional[str]

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str]
    chat_history: List[ChatMessage]
    bmi: Optional[dict] = None

class ChatResponse(BaseModel):
    ai_message: str

class BMIRequest(BaseModel):
    height_cm: float
    weight_kg: float

class BMIResponse(BaseModel):
    bmi: float
    category: str

@app.post("/bmi", response_model=BMIResponse)
def bmi_endpoint(req: BMIRequest):
    result = calculate_bmi(req.height_cm, req.weight_kg)
    return BMIResponse(bmi=result["bmi"], category=result["category"])

@traceable
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request: session_id={request.session_id}")
    # Prepare chat history for LangGraph
    history = []
    for m in request.chat_history:
        if m.role == "user":
            history.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            history.append(AIMessage(content=m.content))
    # Add system prompt, append BMI if provided
    sys_prompt = SYSTEM_PROMPT
    if request.bmi and "bmi" in request.bmi and "category" in request.bmi:
        sys_prompt += f"\n\nUser's BMI provided is {request.bmi['bmi']:.2f} ({request.bmi['category']})."
    # Do not insert SystemMessage here, pass as system_prompt in context
    context = {"messages": history, "system_prompt": sys_prompt}
    response = graph.invoke(context)
    # for m in response['messages']:
    #     logger.info(f" {getattr(m, 'content', str(m))}")
    ai_msgs = [m for m in response["messages"] if hasattr(m, "content")]
    last_ai_msg = ai_msgs[-1].content if ai_msgs else ""
    return ChatResponse(ai_message=last_ai_msg)
