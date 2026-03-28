import json
import logging
import os
import time
import requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL = "openai/gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base/preprocessed"
UNKNOWN_QUESTIONS_PATH = Path(__file__).parent / "unknown_questions.json"
USER_DETAILS_PATH = Path(__file__).parent / "user_details.json"

COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 20
FINAL_K = 10

RETRY_WAIT = wait_exponential(multiplier=1, min=10, max=240)
RETRY_STOP = stop_after_attempt(4)

openai_client = OpenAI()
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

SYSTEM_PROMPT = """
You are acting as Kharisma Rizki Wijanarko. You are answering questions on Rizki's website, \
particularly questions related to Rizki's career, background, skills and experience. \
Your responsibility is to represent Rizki for interactions on the website as faithfully as possible. Answer using only the provided context. \
If missing, say you don't know. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, \
even if it's about something trivial or unrelated to career. Send the notification only if the user provided their user details. \
If the user is engaging in discussion, try to steer them towards getting in touch. Before calling record_user_details, you must have all of: (1) their email OR phone number for contact, (2) whether they are reaching out for company/business reasons or personal use, and (3) a short message they want to leave for Rizki. Ask politely for anything missing; only call record_user_details once you have all three. \
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question: \
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Record a question that the AI does not have the information to answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question that couldn't be answered."
                    },
                    "user_details": {
                        "type": "string",
                        "description": "Any user details known (e.g., name or email) to associate with this question."
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Record a lead for Rizki after the user has shared contact, use context, and a message. Do not call until contact (email or phone), company vs personal, and their message are all known.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact": {
                        "type": "string",
                        "description": "Email address or phone number the user gave for follow-up."
                    },
                    "use_case": {
                        "type": "string",
                        "enum": ["company", "personal"],
                        "description": "company: business, hiring, or organization; personal: individual, networking, or non-corporate."
                    },
                    "message_to_rizki": {
                        "type": "string",
                        "description": "What the user wants Rizki to know or why they are reaching out."
                    },
                    "name": {
                        "type": "string",
                        "description": "User's name if they shared it (optional)."
                    }
                },
                "required": ["contact", "use_case", "message_to_rizki"]
            }
        }
    }
]


# --- Data models ---

class Chunk(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# --- Notifications & persistence ---

def send_telegram_notification(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram credentials missing. Skipping notification.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")


def _append_to_json(path: Path, entry: dict) -> None:
    """Safely append an entry to a local JSON log file."""
    data = []
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def record_unknown_question(question: str, user_details: str | None = None) -> str:
    """Record a question the AI couldn't answer and optionally notify via Telegram."""
    entry = {"question": question, "timestamp": datetime.now().isoformat()}
    if user_details:
        entry["user_details"] = user_details
    _append_to_json(UNKNOWN_QUESTIONS_PATH, entry)
    if user_details:
        send_telegram_notification(
            f"New Unknown Question: {question}\nUser Details: {user_details}"
        )
    return "Question recorded successfully."


def record_user_details(
    contact: str,
    use_case: str,
    message_to_rizki: str,
    name: str | None = None,
) -> str:
    """Record structured lead details and notify via Telegram."""
    entry = {
        "contact": contact.strip(),
        "use_case": use_case,
        "message_to_rizki": message_to_rizki.strip(),
        "timestamp": datetime.now().isoformat(),
    }
    if name and name.strip():
        entry["name"] = name.strip()
    _append_to_json(USER_DETAILS_PATH, entry)
    use_label = "Company" if use_case == "company" else "Personal"
    lines = [
        "New lead for Rizki",
        f"Contact: {entry['contact']}",
        f"Use: {use_label}",
        f"Message: {entry['message_to_rizki']}",
    ]
    if entry.get("name"):
        lines.insert(1, f"Name: {entry['name']}")
    send_telegram_notification("\n".join(lines))
    return "User details recorded successfully. Rizki will be in touch if needed."


def dispatch_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute a tool call by name and return a response string."""
    if tool_name == "record_unknown_question":
        record_unknown_question(arguments["question"], arguments.get("user_details"))
        return "Question recorded successfully. I will notify Rizki."
    elif tool_name == "record_user_details":
        record_user_details(
            arguments["contact"],
            arguments["use_case"],
            arguments["message_to_rizki"],
            arguments.get("name"),
        )
        return "User details recorded successfully. Rizki will be in touch if needed."
    logger.warning(f"Unknown tool called: {tool_name}")
    return "Tool executed."


# --- RAG pipeline ---

@retry(wait=RETRY_WAIT, stop=RETRY_STOP)
def rerank(question: str, chunks: list[Chunk]) -> list[Chunk]:
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
Rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all chunk ids provided, reranked.
"""
    user_prompt = (
        f"The user has asked:\n\n{question}\n\n"
        "Order all chunks by relevance, most to least. Include all chunk ids.\n\n"
        "Here are the chunks:\n\n"
    )
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=RankOrder,
    )
    order = RankOrder.model_validate_json(response.choices[0].message.content).order
    return [chunks[i - 1] for i in order if 1 <= i <= len(chunks)]


def make_rag_messages(question: str, history: list[dict], chunks: list[Chunk]) -> list[dict]:
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=RETRY_WAIT, stop=RETRY_STOP)
def rewrite_query(question: str, history: list[dict] = []) -> str:
    """Rewrite the user's question into a concise knowledge base search query."""
    message = f"""
You are in a conversation about Kharisma Rizki Wijanarko (Rizki)'s career.
You are about to search a Knowledge Base to answer the user's question.

Conversation history:
{history}

User's current question:
{question}

Respond ONLY with a short, precise query to search the Knowledge Base. Nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


def fetch_context_unranked(question: str) -> list[Chunk]:
    query_embedding = (
        openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
        .data[0]
        .embedding
    )
    results = collection.query(query_embeddings=[query_embedding], n_results=RETRIEVAL_K)
    return [
        Chunk(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def merge_chunks(primary: list[Chunk], secondary: list[Chunk]) -> list[Chunk]:
    seen = {chunk.page_content for chunk in primary}
    return primary + [chunk for chunk in secondary if chunk.page_content not in seen]


def fetch_context(original_question: str) -> list[Chunk]:
    """Dual-query retrieval with query rewriting and reranking."""
    try:
        rewritten_question = rewrite_query(original_question)
        logger.info(f"[RAG] Rewritten query: {rewritten_question!r}")
    except Exception as e:
        logger.warning(f"[RAG] Query rewrite failed ({e}), falling back to original.")
        rewritten_question = original_question

    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    merged = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, merged)
    return reranked[:FINAL_K]


# --- Main entry point ---

@retry(wait=RETRY_WAIT, stop=RETRY_STOP)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Chunk]]:
    """
    Answer a question using RAG. Returns the answer string and the retrieved chunks.
    Handles multi-turn tool calls in a loop until the model produces a final text response.
    """
    t0 = time.time()
    chunks = fetch_context(question)
    logger.info(f"[RAG] Context retrieved in {time.time() - t0:.2f}s ({len(chunks)} chunks)")

    messages = make_rag_messages(question, history, chunks)

    # Tool call loop — handles chained tool calls safely
    for _ in range(5): 
        response = completion(model=MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message

        if not message.tool_calls:
            break

        messages.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            tool_response = dispatch_tool_call(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": tool_response,
            })

    logger.info(f"[RAG] Total answer_question time: {time.time() - t0:.2f}s")
    return message.content, chunks