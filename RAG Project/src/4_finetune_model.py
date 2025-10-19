# src/4_finetune_model.py

import json
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# BitsAndBytesConfig 和 peft 相关导入已注释（6GB显卡专用）
# from peft import get_peft_model, LoraConfig, TaskType

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = BASE_DIR / "data" / "processed" / "instructions" / "instructions_trainval.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "finetuned_model"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CoT Prompt template
COT_PROMPT = (
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Lecturre: {lecture}\n"
    "The detailed solution is : {solution}.\n"
    "The correct answer is: {answer_index}.\n"
    "END"
)

class InstructionExample:
    """Represents a single training example parsed from JSONL data."""
    def __init__(self, instruction, input, output_answer_index, output_explanation, metadata):
        self.question = instruction.split("Choices:")[0].replace("Question:", "").strip()
        self.choices = eval(instruction.split("Choices:")[1].strip()) if "Choices:" in instruction else []
        self.answer_index = output_answer_index
        self.lecture = input.strip() or ""
        self.solution = output_explanation.strip() or ""
        self.metadata = metadata

    def to_prompt(self):
        """Formats the example into a training prompt."""
        choices_str = ", ".join([f"{i}. {choice}" for i, choice in enumerate(self.choices)])
        return COT_PROMPT.format(
            question=self.question,
            choices=choices_str,
            lecture=self.lecture,
            answer_index=self.answer_index,
            solution=self.solution
        )

class ScienceQADataset(Dataset):
    """PyTorch Dataset for ScienceQA instruction data."""
    def __init__(self, data_path: Path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    example = InstructionExample(**data)
                    if example.question and isinstance(example.answer_index, int) and example.answer_index >= 0:
                        self.samples.append(example)
                    else:
                        logger.warning("Skipping sample due to missing question or answer.")
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line.")
        
        logger.info(f"Dataset loaded with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]
        choices_str = ", ".join([f"{i}. {choice}" for i, choice in enumerate(example.choices)])
        full_prompt = (
            f"[Question] {example.question}\n"
            f"[Choices] {choices_str}\n"
            f"[Background] {example.lecture}\n"
            f"The detailed [Explanation] is: {example.solution}.\n"
            f"The correct [Answer] is: {example.answer_index}."
        )
        
        prefix = (
            f"[Question] {example.question}\n"
            f"[Choices] {choices_str}\n"
            f"[Background] {example.lecture}\n"
        )
        
        tokenized = self.tokenizer(full_prompt, truncation=True, padding="max_length", 
                                  max_length=self.max_length, return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze()
        
        prefix_len = len(self.tokenizer(prefix, return_tensors="pt").input_ids[0])
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": labels
        }

def main():
    """Orchestrate the full precision fine-tuning process."""
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 注释掉6GB显卡的QLoRA量化配置
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16
    # )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    model.resize_token_embeddings(len(tokenizer))

    # 注释掉6GB显卡的LoRA配置
    # logger.info("Applying QLoRA...")
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    #     bias="none",
    #     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    # )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    logger.info("Preparing dataset...")
    dataset = ScienceQADataset(DATA_PATH, tokenizer)
    if not dataset:
        raise ValueError("Dataset is empty. Check data path or format.")

    logger.info("Setting up training arguments...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=2,   
        learning_rate=5e-6,              
        num_train_epochs=3,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=30,
        bf16=True,  
        seed=888,
        report_to="none",
        remove_unused_columns=False,
    )

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model and tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()