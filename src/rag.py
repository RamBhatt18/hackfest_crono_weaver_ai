import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import logging
from openai import OpenAI # <-- MODIFIED IMPORT

from src.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------
# Config
# ---------------------------
INDEXED_CSV_PATH = "/app/data/output/indexed_data.csv"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K = 5  # Number of sources to retrieve
OPENAI_MODEL = "gpt-3.5-turbo"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# <-- INITIALIZE THE NEW OPENAI CLIENT -->
client = OpenAI(api_key=OPENAI_API_KEY)
# ----------------------------------------


# ---------------------------
# Helper classes
# ---------------------------
class SourceNode:
    """Represents a source document returned by the RAG engine."""
    def __init__(self, node_id, text=None, metadata=None, score=None):
        self.node_id = node_id
        self.text = text
        self.metadata = metadata or {}
        self.score = score or 1.0

    def get_content(self, metadata_mode="all"):
        return self.text


class ChatEngine:
    """Enterprise-ready RAG engine with GPT-3.5 integration."""
    def __init__(self):
        self.index_df = pd.DataFrame()
        self.embeddings = None
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.load_index()

    def load_index(self):
        """Load indexed CSV and embeddings."""
        if os.path.exists(INDEXED_CSV_PATH):
            try:
                self.index_df = pd.read_csv(INDEXED_CSV_PATH)
                if "embedding_str" in self.index_df.columns:
                    self.embeddings = np.array(self.index_df["embedding_str"].apply(eval).tolist())
                    logger.info(f"Loaded {len(self.index_df)} rows with embeddings")
                else:
                    logger.warning("CSV missing 'embedding_str' column")
                    self.embeddings = None
            except Exception as e:
                logger.error(f"Failed to load indexed CSV: {e}")
                self.index_df = pd.DataFrame()
                self.embeddings = None
        else:
            logger.warning("Indexed CSV not found, starting with empty index")
            self.index_df = pd.DataFrame()
            self.embeddings = None

    def reload_index(self):
        """Reload the CSV index."""
        logger.info("Reloading RAG index...")
        self.load_index()
        logger.info("RAG index reloaded successfully")

    # ---------------------------
    # Retrieval
    # ---------------------------
    def retrieve_sources(self, query: str, top_k=TOP_K) -> List[SourceNode]:
        """Return top-k relevant tickets for a query."""
        if self.index_df.empty or self.embeddings is None:
            return []

        query_emb = self.model.encode([query])[0]
        emb_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        query_norm = query_emb / np.linalg.norm(query_emb)
        scores = np.dot(emb_norm, query_norm)

        top_indices = scores.argsort()[::-1][:top_k]
        sources = []
        for idx in top_indices:
            row = self.index_df.iloc[idx]
            sources.append(SourceNode(
                node_id=row.get("ticket_id", "unknown"),
                text=row.get("body", ""),
                metadata=row.to_dict(),
                score=float(scores[idx])
            ))
        return sources

    # ---------------------------
    # GPT-3.5 integration
    # ---------------------------
    def generate_answer(self, query: str, sources: List[SourceNode]) -> str:
        """Use OpenAI GPT to generate answer using retrieved sources."""
        if not sources:
            return f"No relevant tickets found for query: '{query}'"

        context = ""
        for i, s in enumerate(sources, 1):
            context += f"Source {i} (Ticket ID: {s.node_id}): {s.text}\n"

        prompt = f"""
You are an enterprise support assistant. Use the following ticket sources to answer the user query.
Provide a clear, concise answer and cite relevant sources by Ticket ID.

User Query: {query}

Sources:
{context}

Answer:
"""
        # <-- ENTIRE API CALL SECTION IS MODIFIED -->
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful enterprise support assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            answer_text = response.choices[0].message.content.strip()
            return answer_text
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return f"Error generating answer: {e}"
        # ---------------------------------------------

    # ---------------------------
    # Chat interfaces
    # ---------------------------
    def chat(self, query: str):
        """Synchronous chat (Streamlit)."""
        sources = self.retrieve_sources(query)
        answer_text = self.generate_answer(query, sources)
        return type("Response", (), {"response": answer_text, "source_nodes": sources})()

    async def achat(self, query: str):
        """Async chat (FastAPI)."""
        return self.chat(query)


# ---------------------------
# Global singleton
# ---------------------------
_chat_engine_instance = None

def get_chat_engine():
    global _chat_engine_instance
    if _chat_engine_instance is None:
        _chat_engine_instance = ChatEngine()
    return _chat_engine_instance

def reload_index():
    engine = get_chat_engine()
    engine.reload_index()
