# src/config.py

from pathlib import Path

# Set BASE_DIR to the directory containing this file (src folder)
BASE_DIR = Path(__file__).resolve().parent

# Model and data paths under src
MODEL_PATH = BASE_DIR / "models" / "microsoft_phi2"
PROBLEM_JSON = BASE_DIR / "data" / "problems.json"
PID_SPLIT_JSON = BASE_DIR / "data" / "pid_splits.json"

RETRIEVAL_CORPUS_PATH = BASE_DIR / "data" / "processed" / "retrieval_corpus.jsonl"
VECTOR_DB_PATH = BASE_DIR / "data" / "processed" / "vector_db"

DATA_PATH = BASE_DIR / "data" / "processed" / "instructions" / "instructions_train.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "finetuned_qlora" / "full_trained" 

FINETUNED_MODEL_PATH = BASE_DIR / "models" / "finetuned_model" 
FINETUNED_EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "finetuned_embedding"