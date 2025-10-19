# src/3_vector_database.py

import json
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import nltk
from tqdm import tqdm

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK punkt_tab is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Configuration
BASE_DIR = Path(__file__).parent
RETRIEVAL_CORPUS_PATH = BASE_DIR / "data/processed/retrieval_corpus.jsonl"
VECTOR_DB_PATH = BASE_DIR / "data/processed/vector_db"
FINETUNED_EMBEDDING_MODEL_PATH = BASE_DIR / "models/finetuned_embedding"
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
MAX_TOKENS = 512  # Matches typical embedding model max length (e.g., bge-small)

def load_retrieval_corpus(path, max_tokens=MAX_TOKENS):
    documents = []
    model = SentenceTransformer(str(FINETUNED_EMBEDDING_MODEL_PATH))
    tokenizer = model.tokenizer
    logger.info(f"Loading documents from {path}...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing documents"):
                try:
                    item = json.loads(line.strip())
                    full_text = item.get("content", "").strip().replace("\n", " ")
                    metadata = item.get("metadata", {})
                    if not full_text:
                        continue
                    # Improved separation: look for "Solution:" keyword
                    if "Solution:" in full_text:
                        lecture, solution = full_text.split("Solution:", 1)
                        lecture = lecture.strip()
                        solution = "Solution: " + solution.strip()
                    else:
                        lecture = full_text
                        solution = ""
                    # Tokenize and truncate to max_tokens
                    lecture_tokens = tokenizer.tokenize(lecture)
                    solution_tokens = tokenizer.tokenize(solution) if solution else []
                    if len(lecture_tokens) > max_tokens:
                        lecture_tokens = lecture_tokens[:max_tokens]
                    if len(solution_tokens) > max_tokens:
                        solution_tokens = solution_tokens[:max_tokens]
                    lecture = tokenizer.convert_tokens_to_string(lecture_tokens)
                    solution = tokenizer.convert_tokens_to_string(solution_tokens)
                    # Combine with prefixes
                    page_content = (
                        f"Lecture: {lecture}.\nSolution: {solution}." if solution 
                        else f"Lecture: {lecture}."
                    )
                    documents.append(Document(page_content=page_content, metadata=metadata))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line")
                    continue
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise
    logger.info(f"Loaded {len(documents)} documents")
    return documents

def build_vector_db(documents, output_path, embedding_model_path):
    logger.info("Building vector database with high precision...")
    embeddings = HuggingFaceEmbeddings(
        model_name=str(embedding_model_path),
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # Use HNSW index for faster retrieval
    db = FAISS.from_documents(documents, embeddings, distance_strategy="COSINE")
    db.save_local(str(output_path))

def main():
    try:
        documents = load_retrieval_corpus(RETRIEVAL_CORPUS_PATH)
        if not documents:
            logger.error("No documents loaded. Exiting.")
            return
        build_vector_db(documents, VECTOR_DB_PATH, FINETUNED_EMBEDDING_MODEL_PATH)
        logger.info("Vector database built and saved successfully")
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()