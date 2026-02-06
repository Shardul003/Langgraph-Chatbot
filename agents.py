# agents.py
import os
from groq import Groq
from logger import logger

# -------------------------------------------------
# Model configuration (UPDATED TO LIVE MODEL)
# -------------------------------------------------
# Switched from decommissioned llama3-70b-8192 to llama-3.3-70b-versatile
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)


# -------------------------------------------------
# Input Filter Agent
# -------------------------------------------------
def input_filter(state):
    logger.info("Input filter")
    return {
        "query": state["query"],
        "context": [],
        "sources": []
    }


# -------------------------------------------------
# Retriever Agent
# -------------------------------------------------
def retriever(state, vectorstore):
    logger.info("Retriever")

    # k=4 remains standard; ensure vectorstore is pre-loaded via cache_resource in app.py
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


# -------------------------------------------------
# Answer Generation Agent (WITH ERROR HANDLING)
# -------------------------------------------------
def answer_agent(state):
    logger.info("Answer agent")

    try:
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
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        state["answer"] = response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Error in Answer Agent: {e}")
        # Providing a fallback answer prevents the LangGraph execution from crashing the UI
        state["answer"] = f"Error: Unable to generate an answer at this time. (Details: {str(e)})"
    
    return state


# -------------------------------------------------
# Evaluation Agent (WITH ERROR HANDLING)
# -------------------------------------------------
def evaluation_agent(state):
    logger.info("Evaluation agent")

    try:
        client = get_groq_client()

        prompt = f"""
Question: {state["query"]}
Answer: {state["answer"]}

Is the answer grounded in the context?
Reply only YES or NO.
"""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        state["evaluation"] = response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Error in Evaluation Agent: {e}")
        state["evaluation"] = "ERROR"
        
    return state
