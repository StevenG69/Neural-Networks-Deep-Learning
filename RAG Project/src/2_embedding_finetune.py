# src/2_embedding_finetune.py

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import json
import random
from pathlib import Path
import logging

# Configuration
BASE_DIR = Path(__file__).parent
RETRIEVAL_CORPUS_PATH = BASE_DIR / "data/processed/retrieval_corpus.jsonl"
FINETUNED_MODEL_PATH = BASE_DIR / "models/finetuned_embedding"
FINETUNED_MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_instructions(path):
    instructions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            instructions.append(json.loads(line.strip()))
    return instructions

def load_retrieval_corpus(path):
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            corpus[item["question_id"]] = item["content"]
    return corpus

def main():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    logger.info("Loaded base model")

    # Training data
    train_instructions = load_instructions(BASE_DIR / "data/processed/instructions/instructions_trainval.jsonl")
    retrieval_corpus = load_retrieval_corpus(RETRIEVAL_CORPUS_PATH)
    train_examples = [
        InputExample(texts=[inst["instruction"], retrieval_corpus[inst["metadata"]["problem_id"]]])
        for inst in train_instructions
        if inst["metadata"]["problem_id"] in retrieval_corpus
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    logger.info(f"Generated {len(train_examples)} training examples")

    # Validation data
    val_instructions = load_instructions(BASE_DIR / "data/processed/instructions/instructions_test.jsonl")
    val_positive = [
        InputExample(
            texts=[inst["instruction"], retrieval_corpus[inst["metadata"]["problem_id"]]],
            label=1
        )
        for inst in val_instructions
        if inst["metadata"]["problem_id"] in retrieval_corpus
    ]
    val_negative = []
    corpus_contents = list(retrieval_corpus.values())
    for inst in val_instructions:
        pid = inst["metadata"]["problem_id"]
        if pid in retrieval_corpus:
            neg_content = random.choice([c for c in corpus_contents if c != retrieval_corpus[pid]])
            val_negative.append(InputExample(texts=[inst["instruction"], neg_content], label=0))
    val_examples = val_positive + val_negative
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='val')
    logger.info(f"Generated {len(val_examples)} validation examples")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=3,
        warmup_steps=100,
        evaluation_steps=100,
        save_best_model=True,
        output_path=str(FINETUNED_MODEL_PATH),
        optimizer_params={'lr': 2e-5},
        use_amp=True,
    )

    logger.info(f"Model saved to {FINETUNED_MODEL_PATH}")

if __name__ == "__main__":
    main()