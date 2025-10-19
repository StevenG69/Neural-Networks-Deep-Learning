# src/1_data_preprocess.py

import json
from pathlib import Path
import logging

# Configuration
BASE_DIR = Path(__file__).parent
PROBLEM_JSON = BASE_DIR / "data/problems.json"
PID_SPLIT_JSON = BASE_DIR / "data/pid_splits.json"
OUTPUT_DIR = BASE_DIR / "data/processed/instructions"
RETRIEVAL_PATH = BASE_DIR / "data/processed/retrieval_corpus.jsonl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(problem_path, split_path):
    with open(problem_path, "r", encoding="utf-8") as f:
        problems = json.load(f)
        for pid, problem in problems.items():
            problem["id"] = pid
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    return problems, splits

def determine_grade_level(grade_str):
    try:
        grade_num = int(grade_str.replace("grade", ""))
        return "grade1-6" if 1 <= grade_num <= 6 else "grade7-12"
    except ValueError:
        return "unknown"

def convert_to_instruction(problem):
    question = problem["question"]
    choices = problem["choices"]
    answer_idx = problem["answer"]
    lecture = problem.get("lecture", "").strip()
    solution = problem.get("solution", "").strip()
    subject = problem.get("subject", "")
    raw_grade = problem.get("grade", "")
    
    return {
        "instruction": f"Answer the following question and explain reasons:\nQuestion: {question}\nChoices: {choices}",
        "input": lecture,
        "output_answer_index": answer_idx,
        "output_explanation": solution.replace("\n", " ").strip(),
        "metadata": {
            "problem_id": problem["id"],
            "answer_idx": answer_idx,
            "answer_text": choices[answer_idx] if answer_idx < len(choices) else "",
            "subject": subject,
            "grade": raw_grade,
            "grade_level_group": determine_grade_level(raw_grade)
        }
    }

def convert_to_retrieval(problem):
    # Split lecture and solution and prefix for RAG retrieval
    lecture = problem.get("lecture", "").strip()
    solution = problem.get("solution", "").strip()
    # Combine with explicit prefixes
    content = (
        f"Lecture: {lecture}.\n"
        f"Solution: {solution}."
    )
    return {
        "question_id": problem.get("id", "unknown"),
        "content": content,
        "metadata": {
            "subject": problem.get("subject", ""),
            "topic": problem.get("topic", ""),
            "category": problem.get("category", ""),
            "skill": problem.get("skill", ""),
            "grade": problem.get("grade", "")
        }
    }


def main():
    problems, splits = load_data(PROBLEM_JSON, PID_SPLIT_JSON)
    output_files = {
        "train": OUTPUT_DIR / "instructions_train.jsonl",
        "val": OUTPUT_DIR / "instructions_val.jsonl",
        "test": OUTPUT_DIR / "instructions_test.jsonl",
        "minitrain": OUTPUT_DIR / "instructions_minitrain.jsonl",
        "minival": OUTPUT_DIR / "instructions_minival.jsonl",
        "minitest": OUTPUT_DIR / "instructions_minitest.jsonl",
        "all": OUTPUT_DIR / "instructions_all.jsonl",
        "trainval": OUTPUT_DIR / "instructions_trainval.jsonl",
        "minitrainval": OUTPUT_DIR / "instructions_minitrainval.jsonl"
    }

    # Initialize file handles and data collections
    with ExitStack() as stack:
        files = {key: stack.enter_context(open(path, "w", encoding="utf-8")) 
                 for key, path in output_files.items()}
        retrieval_file = stack.enter_context(open(RETRIEVAL_PATH, "w", encoding="utf-8"))
        written_ids = set()
        retrieval_written_ids = set()
        trainval_data = []
        minitrainval_data = []

        for split_name, ids in splits.items():
            for pid in ids:
                problem = problems.get(pid)
                if not problem or problem.get("image") is not None:
                    continue
                inst_data = convert_to_instruction(problem)
                
                # Write to split-specific file
                if split_name in files:
                    files[split_name].write(json.dumps(inst_data, ensure_ascii=False) + "\n")
                
                # Write to 'all' file if not already written
                if pid not in written_ids:
                    files["all"].write(json.dumps(inst_data, ensure_ascii=False) + "\n")
                    written_ids.add(pid)
                
                # Write to retrieval corpus if not already written
                if pid not in retrieval_written_ids:
                    ret_data = convert_to_retrieval(problem)
                    retrieval_file.write(json.dumps(ret_data, ensure_ascii=False) + "\n")
                    retrieval_written_ids.add(pid)
                
                # Collect trainval and minitrainval data
                if split_name in ["train", "val"]:
                    trainval_data.append(inst_data)
                if split_name in ["minitrain", "minival"]:
                    minitrainval_data.append(inst_data)

        # Write trainval and minitrainval files
        for data, path in [(trainval_data, output_files["trainval"]), 
                           (minitrainval_data, output_files["minitrainval"])]:
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Instructions saved to:")
    for key, path in output_files.items():
        logger.info(f"  - {key}: {path}")
    logger.info(f"Retrieval corpus saved to: {RETRIEVAL_PATH}")

if __name__ == "__main__":
    from contextlib import ExitStack
    main()