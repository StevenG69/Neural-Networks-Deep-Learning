import json
import logging
import re
from pathlib import Path
import argparse
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
import rag_chain
import numpy as np  # 新增：用于计算标准差
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # 新增：用于BLEU评分

nltk.download('punkt_tab')

# Configuration
BASE_DIR = Path(__file__).parent
FINETUNED_EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "finetuned_embedding"
FINETUNED_MODEL_PATH = BASE_DIR / "models" / "finetuned_model"
VECTOR_DB_PATH = BASE_DIR / "data" / "processed" / "vector_db"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "instructions" / "instructions_test.jsonl"
PROBLEM_JSON = BASE_DIR / "data" / "problems.json"
OUTPUT_DIR = BASE_DIR / "models" / "finetuned_model" / "results"
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load semantic similarity model
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
# Simple answer index extraction function for direct predictions
def extract_choice_index_simple(text, num_choices):
    patterns = [r"^\s*(\d+)", r"[Aa]nswer:?\s*(\d+)", r"^\s*([A-Ea-e])", r"[Aa]nswer:?\s*([A-Ea-e])"]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1)
            if val.isdigit() and 0 <= int(val) < num_choices:
                return int(val)
            if val.upper() in 'ABCDE':
                idx = ord(val.upper()) - ord('A')
                if 0 <= idx < num_choices:
                    return idx
    return None
# Load answer mapping
def load_answer_mapping(problem_json_path):
    answer_mapping = {}
    with open(problem_json_path, "r", encoding="utf-8") as f:
        problems = json.load(f)
        for pid, problem in problems.items():
            answer_index = problem.get("answer", -1)
            if answer_index != -1:
                answer_mapping[pid] = answer_index
    logger.info(f"Loaded answer mapping for {len(answer_mapping)} problems.")
    return answer_mapping
# Load test dataset
def load_test_dataset(test_data_path, answer_mapping, limit=None):
    test_data = []
    with open(PROBLEM_JSON, "r", encoding="utf-8") as f:
        problems = json.load(f)
    with open(test_data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading test dataset")):
            if limit is not None and idx >= limit:
                break
            item = json.loads(line.strip())
            problem_id = item.get("metadata", {}).get("problem_id", "unknown")
            instruction = item.get("instruction", "")
            choices = rag_chain.parse_choices(instruction.split("Choices:")[1].strip()) if "Choices:" in instruction else []
            item["metadata"]["choices"] = choices
            item["output_answer_index"] = answer_mapping.get(problem_id, -1)
            problem_data = problems.get(problem_id, {})
            item["metadata"]["solution"] = problem_data.get("solution", "")
            item["metadata"]["lecture"] = problem_data.get("lecture", "")
            grade = problem_data.get("grade", "").lower()
            item["metadata"]["grade_group"] = "grade1-6" if grade in [f"grade{i}" for i in range(1, 7)] else "grade7-12"
            item["metadata"]["subject"] = problem_data.get("subject", "unknown").lower()
            test_data.append(item)
    logger.info(f"Loaded {len(test_data)} test items.")
    return test_data
# Unified prediction generation function
def generate_predictions(model, tokenizer, dataset, db=None, batch_size=24, is_rag=False):
    pred_indices, ref_indices, pred_texts, ref_texts = [], [], [], []
    grade_groups = {"grade1-6": [], "grade7-12": []}
    subjects = {"language science": [], "natural science": [], "social science": []}
    prompts_list = []  # Added to record prompts
    outputs_list = []  # Added to record outputs
    model.eval()
    device = next(model.parameters()).device
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    for batch in tqdm(batches, desc=f"{'RAG' if is_rag else 'Direct'} Predictions"):
        prompts, choices_list, ref_idx_list = [], [], []
        grade_group_list, subject_list = [], []
        # Prepare batch data
        for item in batch:
            instruction = item.get("instruction", "")
            ref_idx = item.get("output_answer_index", -1)
            metadata = item.get("metadata", {})
            choices = metadata.get("choices", [])
            question = instruction.split("Choices:")[0].split("Question:")[1].strip() if "Choices:" in instruction else ""
            choices_str = ", ".join([f"{i}. {c}" for i, c in enumerate(choices)])
            if is_rag:
                query = f"{question} {' '.join(choices)}"
                context = rag_chain.retrieve_context(db, query, k=3)
                prompt = rag_chain.build_cot_prompt_for_inference(question, choices, context)
            else:
                prompt = f"Question: {question}\nChoices: {choices_str}\nSelect the correct answer index (0-{len(choices)-1}):"
            prompts.append(prompt)
            choices_list.append(choices)
            ref_idx_list.append(ref_idx)
            grade_group_list.append(metadata.get("grade_group", "unknown"))
            subject_list.append(metadata.get("subject", "unknown"))
        # Generate answers
        outputs = rag_chain.generate_answer(model, tokenizer, prompts, max_new_tokens=100 if is_rag else 50, batch=True)
        # Extract answers and record prompts/outputs
        for prompt, output, choices, ref_idx, grade_group, subject, item in zip(prompts, outputs, choices_list, ref_idx_list, grade_group_list, subject_list, batch):
            if is_rag:
                result = rag_chain.extract_answer(output, "", len(choices)) if choices else {"index": None, "text": output}
                pred_idx = result.get("index")
                pred_text = result.get("text", output)
            else:
                pred_idx = extract_choice_index_simple(output, len(choices)) if choices else None
                pred_text = output
            pred_indices.append(pred_idx)
            ref_indices.append(ref_idx)
            pred_texts.append(pred_text)
            ref_texts.append(item["metadata"]["lecture"] + "" + item["metadata"]["solution"])
            if grade_group in grade_groups:
                grade_groups[grade_group].append((pred_idx, ref_idx, pred_text, item["metadata"]["lecture"] + "" + item["metadata"]["solution"]))
            if subject in subjects:
                subjects[subject].append((pred_idx, ref_idx, pred_text, item["metadata"]["lecture"] + "" + item["metadata"]["solution"]))
            prompts_list.append(prompt)
            outputs_list.append(output)
    return pred_indices, ref_indices, pred_texts, ref_texts, grade_groups, subjects, prompts_list, outputs_list
# Compute grouped metrics
def compute_grouped_metrics(pred_indices, ref_indices, grade_groups, subjects):
    metrics = {
        "overall": {"Accuracy": 0.0, "Precision_Weighted": 0.0, "Recall_Weighted": 0.0, "F1_Weighted": 0.0, "Success_Rate": 0.0},
        "grade_groups": {"Grade 1-6": {}, "Grade 7-12": {}},
        "subjects": {"Language Science": {}, "Natural Science": {}, "Social Science": {}}
    }
    # Overall metrics
    valid_pairs = [(p, r) for p, r in zip(pred_indices, ref_indices) if p is not None and isinstance(p, int) and 0 <= p]
    if valid_pairs:
        valid_preds, valid_refs = zip(*valid_pairs)
        num_classes = max(max(valid_refs, default=0), max(valid_preds, default=0)) + 1
        labels = list(range(num_classes))
        metrics["overall"] = {
            "Accuracy": accuracy_score(valid_refs, valid_preds),
            "Precision_Weighted": precision_score(valid_refs, valid_preds, average="weighted", zero_division=0, labels=labels),
            "Recall_Weighted": recall_score(valid_refs, valid_preds, average="weighted", zero_division=0, labels=labels),
            "F1_Weighted": f1_score(valid_refs, valid_preds, average="weighted", zero_division=0, labels=labels),
            "Success_Rate": len(valid_pairs) / len(ref_indices) if ref_indices else 0
        }
    # Grade group metrics
    for group in grade_groups:
        valid_pairs = [(p, r) for p, r, _, _ in grade_groups[group] if p is not None and isinstance(p, int) and 0 <= p]
        if valid_pairs:
            valid_preds, valid_refs = zip(*valid_pairs)
            num_classes = max(max(valid_refs, default=0), max(valid_preds, default=0)) + 1
            labels = list(range(num_classes))
            metrics["grade_groups"][group] = {
                "Accuracy": accuracy_score(valid_refs, valid_preds),
                "Precision_Weighted": precision_score(valid_refs, valid_preds, average="weighted", zero_division=0, labels=labels),
                "Recall_Weighted": recall_score(valid_refs, valid_preds, average="weighted", zero_division=0, labels=labels),
                "F1_Weighted": f1_score(valid_refs, valid_preds, average="weighted", zero_division=0, labels=labels)
            }
        else:
            metrics["grade_groups"][group] = {"Accuracy": 0.0, "Precision_Weighted": 0.0, "Recall_Weighted": 0.0, "F1_Weighted": 0.0}
    # Subject metrics
    for subject in subjects:
        valid_pairs = [(p, r) for p, r, _, _ in subjects[subject] if p is not None and isinstance(p, int) and 0 <= p]
        if valid_pairs:
            valid_preds, valid_refs = zip(*valid_pairs)
            num_classes = max(max(valid_refs, default=0), max(valid_preds, default=0)) + 1
            labels = list(range(num_classes))
            metrics["subjects"][subject] = {
                "Accuracy": accuracy_score(valid_refs, valid_preds),
                "Precision_Weighted": precision_score(valid_refs, valid_preds, average="weighted", zero_division=0, labels=labels),
                "Recall_Weighted": recall_score(valid_refs, valid_preds, average="weighted", zero_division=0, labels=labels),
                "F1_Weighted": f1_score(valid_refs, valid_preds, average="weighted", zero_division=0, labels=labels)
            }
        else:
            metrics["subjects"][subject] = {"Accuracy": 0.0, "Precision_Weighted": 0.0, "Recall_Weighted": 0.0, "F1_Weighted": 0.0}
    return metrics

# Compute semantic similarity (modified to output mean and std)
def compute_similarity(pred_texts, ref_texts, grade_groups, subjects):
    similarities = {
        "overall": {"mean": 0.0, "std": 0.0},
        "grade_groups": {"grade1-6": {"mean": 0.0, "std": 0.0}, "grade7-12": {"mean": 0.0, "std": 0.0}},
        "subjects": {"language science": {"mean": 0.0, "std": 0.0}, "natural science": {"mean": 0.0, "std": 0.0}, "social science": {"mean": 0.0, "std": 0.0}}
    }
    # Overall similarity
    valid_pairs = [(p, r) for p, r in zip(pred_texts, ref_texts) if p.strip() and r.strip()]
    if valid_pairs:
        pred_embeddings = similarity_model.encode([p for p, _ in valid_pairs], convert_to_tensor=True)
        ref_embeddings = similarity_model.encode([r for _, r in valid_pairs], convert_to_tensor=True)
        sim_scores = util.pytorch_cos_sim(pred_embeddings, ref_embeddings).diagonal().cpu().numpy()
        similarities["overall"] = {
            "mean": float(np.mean(sim_scores)) if sim_scores.size > 0 else 0.0,
            "std": float(np.std(sim_scores)) if sim_scores.size > 0 else 0.0
        }
    # Grade group similarity
    for group in grade_groups:
        valid_pairs = [(p, r) for _, _, p, r in grade_groups[group] if p.strip() and r.strip()]
        if valid_pairs:
            pred_embeddings = similarity_model.encode([p for p, _ in valid_pairs], convert_to_tensor=True)
            ref_embeddings = similarity_model.encode([r for _, r in valid_pairs], convert_to_tensor=True)
            sim_scores = util.pytorch_cos_sim(pred_embeddings, ref_embeddings).diagonal().cpu().numpy()
            similarities["grade_groups"][group] = {
                "mean": float(np.mean(sim_scores)) if sim_scores.size > 0 else 0.0,
                "std": float(np.std(sim_scores)) if sim_scores.size > 0 else 0.0
            }
    # Subject similarity
    for subject in subjects:
        valid_pairs = [(p, r) for _, _, p, r in subjects[subject] if p.strip() and r.strip()]
        if valid_pairs:
            pred_embeddings = similarity_model.encode([p for p, _ in valid_pairs], convert_to_tensor=True)
            ref_embeddings = similarity_model.encode([r for _, r in valid_pairs], convert_to_tensor=True)
            sim_scores = util.pytorch_cos_sim(pred_embeddings, ref_embeddings).diagonal().cpu().numpy()
            similarities["subjects"][subject] = {
                "mean": float(np.mean(sim_scores)) if sim_scores.size > 0 else 0.0,
                "std": float(np.std(sim_scores)) if sim_scores.size > 0 else 0.0
            }
    return similarities

# Compute BLEU score (only norm1, norm2)
def compute_bleu(pred_texts, ref_texts, grade_groups, subjects):
    bleu_scores = {
        "overall": {"bleu1": 0.0, "bleu2": 0.0},
        "grade_groups": {"grade1-6": {"bleu1": 0.0, "bleu2": 0.0}, "grade7-12": {"bleu1": 0.0, "bleu2": 0.0}},
        "subjects": {
            "language science": {"bleu1": 0.0, "bleu2": 0.0},
            "natural science": {"bleu1": 0.0, "bleu2": 0.0},
            "social science": {"bleu1": 0.0, "bleu2": 0.0}
        }
    }
    
    # 设置平滑函数 (norm1)
    smoothing = SmoothingFunction().method1
    
    # Overall BLEU
    valid_pairs = [(p, r) for p, r in zip(pred_texts, ref_texts) if p.strip() and r.strip()]
    if valid_pairs:
        bleu1_scores = []
        bleu2_scores = []
        for pred, ref in valid_pairs:
            # 将文本分割成单词
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # 计算BLEU-1
            bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1,), smoothing_function=smoothing)
            bleu1_scores.append(bleu1)
            
            # 计算BLEU-2
            if len(pred_tokens) >= 2 and len(ref_tokens) >= 2:
                bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5), smoothing_function=smoothing)
                bleu2_scores.append(bleu2)
        
        bleu_scores["overall"] = {
            "bleu1": np.mean(bleu1_scores) if bleu1_scores else 0.0,
            "bleu2": np.mean(bleu2_scores) if bleu2_scores else 0.0
        }
    
    # Grade group BLEU
    for group in grade_groups:
        valid_pairs = [(p, r) for _, _, p, r in grade_groups[group] if p.strip() and r.strip()]
        if valid_pairs:
            bleu1_scores = []
            bleu2_scores = []
            for pred, ref in valid_pairs:
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1,), smoothing_function=smoothing)
                bleu1_scores.append(bleu1)
                
                if len(pred_tokens) >= 2 and len(ref_tokens) >= 2:
                    bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5), smoothing_function=smoothing)
                    bleu2_scores.append(bleu2)
            
            bleu_scores["grade_groups"][group] = {
                "bleu1": np.mean(bleu1_scores) if bleu1_scores else 0.0,
                "bleu2": np.mean(bleu2_scores) if bleu2_scores else 0.0
            }
    
    # Subject BLEU
    for subject in subjects:
        valid_pairs = [(p, r) for _, _, p, r in subjects[subject] if p.strip() and r.strip()]
        if valid_pairs:
            bleu1_scores = []
            bleu2_scores = []
            for pred, ref in valid_pairs:
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1,), smoothing_function=smoothing)
                bleu1_scores.append(bleu1)
                
                if len(pred_tokens) >= 2 and len(ref_tokens) >= 2:
                    bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5), smoothing_function=smoothing)
                    bleu2_scores.append(bleu2)
            
            bleu_scores["subjects"][subject] = {
                "bleu1": np.mean(bleu1_scores) if bleu1_scores else 0.0,
                "bleu2": np.mean(bleu2_scores) if bleu2_scores else 0.0
            }
    
    return bleu_scores

# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek-R1-Distill-Qwen-1.5B models on ScienceQA dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit test samples (e.g., 100 for quick testing).")
    args = parser.parse_args()
    logger.info("Starting Evaluation...")
    answer_mapping = load_answer_mapping(PROBLEM_JSON)
    test_data = load_test_dataset(TEST_DATA_PATH, answer_mapping, limit=args.limit)
    db = rag_chain.load_vector_db(VECTOR_DB_PATH, FINETUNED_EMBEDDING_MODEL_PATH)
    model, tokenizer = rag_chain.load_model_and_tokenizer(FINETUNED_MODEL_PATH)
    
    # Generate predictions (只运行RAG部分)
    logger.info("Generating RAG predictions...")
    rag_results = generate_predictions(model, tokenizer, test_data, db, is_rag=True)
    
    # Compute metrics and similarity
    rag_metrics = compute_grouped_metrics(*rag_results[:2], rag_results[4], rag_results[5])
    rag_similarity = compute_similarity(*rag_results[2:4], rag_results[4], rag_results[5])
    rag_bleu = compute_bleu(*rag_results[2:4], rag_results[4], rag_results[5])  # 新增：计算BLEU分数
    
    # Output results
    print("\nOverall Metrics:")
    print(f"{'Metric':<25} | {'RAG + Trained':<20}")
    print("-" * 50)
    for metric in ["Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted", "Success_Rate"]:
        rag_val = f"{rag_metrics.get('overall', {}).get(metric, 'N/A'):.4f}"
        print(f"{metric:<25} | {rag_val:<20}")
    print(f"{'Semantic Similarity (mean)':<25} | {rag_similarity['overall']['mean']:.4f}")
    print(f"{'Semantic Similarity (std)':<25} | {rag_similarity['overall']['std']:.4f}")
    print(f"{'BLEU-1':<25} | {rag_bleu['overall']['bleu1']:.4f}")
    print(f"{'BLEU-2':<25} | {rag_bleu['overall']['bleu2']:.4f}")
    
    print("\nGrade Group Metrics:")
    for group in ["grade1-6", "grade7-12"]:
        print(f"\n{group}:")
        print(f"{'Metric':<25} | {'RAG + Trained':<20}")
        print("-" * 50)
        for metric in ["Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted"]:
            rag_val = f"{rag_metrics.get('grade_groups', {}).get(group, {}).get(metric, 'N/A'):.4f}"
            print(f"{metric:<25} | {rag_val:<20}")
        print(f"{'Semantic Similarity (mean)':<25} | {rag_similarity['grade_groups'][group]['mean']:.4f}")
        print(f"{'Semantic Similarity (std)':<25} | {rag_similarity['grade_groups'][group]['std']:.4f}")
        print(f"{'BLEU-1':<25} | {rag_bleu['grade_groups'][group]['bleu1']:.4f}")
        print(f"{'BLEU-2':<25} | {rag_bleu['grade_groups'][group]['bleu2']:.4f}")
    
    print("\nSubject Metrics:")
    for subject in ["language science", "natural science", "social science"]:
        print(f"\n{subject}:")
        print(f"{'Metric':<25} | {'RAG + Trained':<20}")
        print("-" * 50)
        for metric in ["Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted"]:
            rag_val = f"{rag_metrics.get('subjects', {}).get(subject, {}).get(metric, 'N/A'):.4f}"
            print(f"{metric:<25} | {rag_val:<20}")
        print(f"{'Semantic Similarity (mean)':<25} | {rag_similarity['subjects'][subject]['mean']:.4f}")
        print(f"{'Semantic Similarity (std)':<25} | {rag_similarity['subjects'][subject]['std']:.4f}")
        print(f"{'BLEU-1':<25} | {rag_bleu['subjects'][subject]['bleu1']:.4f}")
        print(f"{'BLEU-2':<25} | {rag_bleu['subjects'][subject]['bleu2']:.4f}")
    
    # Save detailed prompts and outputs to JSON files
    def save_details_to_json(test_data, prompts_list, outputs_list, model_name):
        details = []
        for item, prompt, output in zip(test_data, prompts_list, outputs_list):
            details.append({
                "problem_id": item["metadata"]["problem_id"],
                "prompt": prompt,
                "output": output,
                "metadata": item["metadata"]
            })
        details.sort(key=lambda x: x["problem_id"])  # Sort by problem_id
        output_path = OUTPUT_DIR / f"{model_name}_details.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        logger.info(f"{model_name} details saved to {output_path}")
    
    save_details_to_json(test_data, rag_results[6], rag_results[7], "rag_trained")
    
    # Save results
    results = {
        "RAG_Metrics": rag_metrics,
        "RAG_Similarity": rag_similarity,
        "RAG_BLEU": rag_bleu,  # 新增：保存BLEU分数
        "Details": {
            "RAG_Predictions_Indices": rag_results[0],
            "RAG_References_Indices": rag_results[1],
            "RAG_Grade_Groups": {k: [(p, r, pt, rt) for p, r, pt, rt in v] for k, v in rag_results[4].items()},
            "RAG_Subjects": {k: [(p, r, pt, rt) for p, r, pt, rt in v] for k, v in rag_results[5].items()}
        }
    }
    results_path = OUTPUT_DIR / "evaluation_results.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_path}")
if __name__ == "__main__":
    main()