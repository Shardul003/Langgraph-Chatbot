# agents.py
import os
from groq import Groq
from logger import logger


# ---------- SAFE GROQ CLIENT ----------
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Check .env or environment variables.")
    return Groq(api_key=api_key)


# ---------- INPUT FILTER ----------
def input_filter(state):
    logger.info("Input filter")
    return {
        "query": state["query"],
        "context": [],
        "sources": []
    }


# ---------- RETRIEVER ----------
def retriever(state, vectorstore):
    logger.info("Retriever")
    docs = vectorstore.similarity_search(state["query"], k=4)

    state["context"] = [d.page_content for d in docs]
    state["sources"] = [
        {
            "pdf": d.metadata.get("source"),
            "page": d.metadata.get("page")
        }
        for d in docs
    ]
    return state


# ---------- ANSWER AGENT ----------
def answer_agent(state):
    logger.info("Answer agent")

    client = get_groq_client()

    prompt = f"""
Answer strictly from the context.
If the answer is not present, say "Not found in documents".

Context:
{chr(10).join(state["context"])}

Question:
{state["query"]}
"""

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    state["answer"] = response.choices[0].message.content
    return state


# ---------- EVALUATION AGENT ----------
def evaluation_agent(state):
    logger.info("Evaluation agent")

    client = get_groq_client()

    prompt = f"""
Question: {state["query"]}
Answer: {state["answer"]}

Is the answer grounded in the context?
Reply with only YES or NO.
"""

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    state["evaluation"] = response.choices[0].message.content.strip()
    return state