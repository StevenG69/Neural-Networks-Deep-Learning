# src/2.5_embedding_ft_evaluate.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
import logging
from pathlib import Path

# 配置
BASE_DIR = Path(__file__).parent
RETRIEVAL_CORPUS_PATH = BASE_DIR / "data/processed/retrieval_corpus.jsonl"
INSTRUCTIONS_PATH = BASE_DIR / "data/processed/instructions/instructions_test.jsonl"
FINETUNED_MODEL_PATH = BASE_DIR / "models/finetuned_embedding"
EVALUATION_OUTPUT_PATH = FINETUNED_MODEL_PATH / "evaluation"
EVALUATION_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
TOP_K_FOR_EVAL = 5

# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_retrieval_corpus(path):
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            corpus[item["question_id"]] = item["content"]
    return corpus

def load_instructions(path):
    instructions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            instructions.append(json.loads(line.strip()))
    return instructions

def evaluate_retrieval(model, questions, corpus_embeddings, corpus_ids, k=TOP_K_FOR_EVAL):
    logger.info("Generating embeddings for questions...")
    question_embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
    sim_matrix = cosine_similarity(question_embeddings, corpus_embeddings)
    
    mrr_sum, total_queries = 0, len(questions)
    hit_at_k = 0
    precisions_at_k, recalls_at_k = [], []

    logger.info(f"Calculating retrieval metrics for {total_queries} queries...")
    for i in range(total_queries):
        true_pid = questions[i]["metadata"]["problem_id"]
        sim_scores = sim_matrix[i]
        sorted_indices = np.argsort(sim_scores)[::-1]
        sorted_pids = [corpus_ids[idx] for idx in sorted_indices]

        for rank, pid in enumerate(sorted_pids, 1):
            if pid == true_pid:
                mrr_sum += 1 / rank
                break

        if true_pid in sorted_pids[:k]:
            hit_at_k += 1

        top_k_pids = sorted_pids[:k]
        relevant = 1 if true_pid in top_k_pids else 0
        precisions_at_k.append(relevant / k)
        recalls_at_k.append(relevant) 

    mrr = mrr_sum / total_queries if total_queries else 0.0
    hit_rate = hit_at_k / total_queries if total_queries else 0.0
    avg_precision = np.mean(precisions_at_k) if precisions_at_k else 0.0
    avg_recall = np.mean(recalls_at_k) if recalls_at_k else 0.0

    return mrr, hit_rate, avg_precision, avg_recall

def visualize_embeddings_umap_3d(question_embeddings, corpus_embeddings, question_labels, corpus_labels, filename="umap_scienceqa_3d.png"):
    logger.info("Performing UMAP 3D dimensionality reduction...")
    all_embeddings = np.vstack([question_embeddings, corpus_embeddings])
    all_labels = question_labels + corpus_labels
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', n_components=3, random_state=42)
    try:
        embedding_3d = reducer.fit_transform(all_embeddings)
        logger.info("Creating 3D UMAP plot...")

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.Spectral(np.linspace(0, 1, 2))
        ax.scatter(embedding_3d[:len(question_embeddings), 0], embedding_3d[:len(question_embeddings), 1], embedding_3d[:len(question_embeddings), 2],
                   c=colors[0], label="Questions", s=5, alpha=0.7)
        ax.scatter(embedding_3d[len(question_embeddings):, 0], embedding_3d[len(question_embeddings):, 1], embedding_3d[len(question_embeddings):, 2],
                   c=colors[1], label="Contents", s=5, alpha=0.7)

        ax.set_title("UMAP 3D Visualization of Questions and Contents")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.legend()
        plt.tight_layout()
        plt.savefig(EVALUATION_OUTPUT_PATH / filename, dpi=300)
        plt.close()
        logger.info(f"3D UMAP plot saved to {EVALUATION_OUTPUT_PATH / filename}")
    except Exception as e:
        logger.error(f"UMAP 3D visualization failed: {e}")

def main():
    model = SentenceTransformer(str(FINETUNED_MODEL_PATH))
    logger.info("Loaded fine-tuned model")

    corpus = load_retrieval_corpus(RETRIEVAL_CORPUS_PATH)
    instructions = load_instructions(INSTRUCTIONS_PATH)
    questions = [inst["instruction"] for inst in instructions]
    true_pids = [inst["metadata"]["problem_id"] for inst in instructions]

    corpus_ids = list(corpus.keys())
    corpus_contents = list(corpus.values())
    corpus_embeddings = model.encode(corpus_contents, convert_to_numpy=True, show_progress_bar=True)

    mrr, hit_rate, avg_precision, avg_recall = evaluate_retrieval(model, instructions, corpus_embeddings, corpus_ids)
    logger.info("--- Retrieval Evaluation Results ---")
    logger.info(f"MRR: {mrr:.4f}")
    logger.info(f"Hit Rate@{TOP_K_FOR_EVAL}: {hit_rate:.4f}")
    logger.info(f"Average Precision@{TOP_K_FOR_EVAL}: {avg_precision:.4f}")
    logger.info(f"Average Recall@{TOP_K_FOR_EVAL}: {avg_recall:.4f}")

    if len(questions) > 100 and len(corpus_contents) > 100:
        logger.info("Generating UMAP 3D visualization...")
        question_embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
        visualize_embeddings_umap_3d(question_embeddings, corpus_embeddings, ["Question"]*len(questions), ["Content"]*len(corpus_contents))

    metrics = {
        "retrieval": {
            "MRR": round(mrr, 4),
            f"HitRate@{TOP_K_FOR_EVAL}": round(hit_rate, 4),
            f"AvgPrecision@{TOP_K_FOR_EVAL}": round(avg_precision, 4),
            f"AvgRecall@{TOP_K_FOR_EVAL}": round(avg_recall, 4)
        }
    }
    metrics_path = EVALUATION_OUTPUT_PATH / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
