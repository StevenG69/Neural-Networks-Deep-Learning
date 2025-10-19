# Neural-Networks-Deep-Learning

# RAG-Enhanced Science QA System for Educational Scenarios

## Project Overview
- **Background & Objectives**: Developed a Retrieval-Augmented Generation (RAG) system based on the K-12 ScienceQA dataset with 21,208 multiple-choice questions. The goal was to enhance explainability and accuracy on resource-constrained small/medium LLMs by integrating RAG with embedding fusion, while validating practicality and reasoning quality for educational deployments.
- **Key Achievements**: Achieved a 7.39% accuracy improvement and 19.55% semantic similarity boost on the fine-tuned DeepSeek-1.5B model compared to baselines. Balanced cost-performance-explainability for real-world educational and commercial applications.
- **Tech Stack**: Python (LangChain, HuggingFace, SentenceTransformers, Gradio); Embeddings (BAAI/bge-small-en-v1.5); LLMs (DeepSeek-1.5B); Vector DB (FAISS); Fine-Tuning (LoRA/Full-Parameter); Evaluation Metrics (Accuracy, BLEU, Semantic Similarity).

## Step-by-Step Development Process

### 1. Data Preprocessing (`1_data_preprocess.py`)
- Loaded ScienceQA problems and splits from JSON files.
- Converted questions into instruction formats: e.g., "Answer the following question and explain reasons: Question: {question} Choices: {choices}".
- Created retrieval corpus by prefixing lecture and solution content (e.g., "Lecture: {lecture}. Solution: {solution}.").
- Split data into train/val/test/mini subsets and saved as JSONL files for fine-tuning and retrieval.
- Ensured metadata inclusion (e.g., problem ID, subject, grade level groups like grade1-6/grade7-12).

### 2. Embedding Model Fine-Tuning (`2_embedding_finetune.py` & `2.5_embedding_ft_evaluate.py`)
- Used base model: BAAI/bge-small-en-v1.5.
- Prepared training examples pairing instructions with corresponding corpus content; validation with positive/negative pairs.
- Fine-tuned using MultipleNegativesRankingLoss for 3 epochs (batch size 16, LR 2e-5).
- Evaluated retrieval: MRR, Hit Rate@5, Avg Precision@5, Avg Recall@5.
- Visualized embeddings with 3D UMAP to assess clustering of questions and contents.
- Saved fine-tuned model for vector DB creation.

### 3. Vector Database Construction (`3_vector_database.py`)
- Loaded retrieval corpus and tokenized/truncated content to max 512 tokens using fine-tuned embedding tokenizer.
- Separated lecture and solution with prefixes for better RAG context.
- Built FAISS vector DB with cosine similarity and HNSW index for efficient retrieval.
- Used HuggingFaceEmbeddings with CUDA for high-precision embedding generation.
- Saved DB for use in RAG pipeline.

### 4. LLM Fine-Tuning (`4_finetune_model.py`)
- Base model: DeepSeek-R1-Distill-Qwen-1.5B.
- Formatted prompts with CoT (Chain-of-Thought): "Question: {question} Choices: {choices} Lecture: {lecture} The detailed solution is: {solution}. The correct answer is: {answer_index}."
- Created dataset with input_ids, attention masks, and labels (ignoring prefix for causal LM).
- Trained with full precision (bfloat16) for 3 epochs (batch size 4, gradient accumulation 2, LR 5e-6).
- (Note: LoRA config commented for 6GB GPU compatibility; focused on full-parameter tuning.)
- Saved fine-tuned model and tokenizer.

### 5. RAG Chain Implementation (`rag_chain.py` & `5_rag_chain.py`)
- Loaded vector DB, fine-tuned embeddings, and LLM.
- Retrieval: Used similarity search (k=3) with sentence tokenization and truncation.
- Prompt Building: Integrated retrieved lecture into CoT template for inference.
- Generation: Batch/single inference with cleaning (remove EOS/stop markers, repetition penalty 1.1).
- Answer Extraction: Regex patterns to parse index and explanation from response.
- Interactive Mode: Command-line interface for testing questions/choices.

### 6. Model Evaluation (`6_evaluate_model.py`)
- Loaded test data with ground-truth answers and metadata.
- Generated predictions with RAG + CoT on fine-tuned model.
- Metrics: Accuracy, Precision/Recall/F1 (weighted), Success Rate; Grouped by grade (1-6/7-12) and subject (language/natural/social science).
- Additional: Semantic similarity (all-MiniLM-L6-v2), BLEU-1/2 scores for explanations.
- Saved detailed prompts/outputs and results JSON (e.g., overall accuracy, per-group stats).
- Compared RAG vs. direct predictions (focused on RAG in final run).

### 7. WebUI Deployment (`7_webui_gradio.py`)
- Built Gradio interface for interactive querying: Inputs (question, choices, optional topic); Outputs (answer index/choice/explanation, context, full response).
- Loaded RAG components on-demand (DB, model, tokenizer).
- UI Features: Markdown formatting, accordions for context/output, footer.
- Launched server (port 7860, shareable) for educators' real-time use.
- Optimized for low-latency with LoRA switching for cost control.

## Product Decisions & Deployment
- Best config: 24.7 min training on A100 (40GB); Balanced cost/performance/explainability.
- Encapsulated full pipeline (retrieval, CoT, inference, eval, logging) into WebUI.
- Supports LoRA model switching for flexible cost/latency control.
- Practical for educational landing: Direct commercial/teaching value.

## Challenges & Learnings
- Handled truncated content and invalid JSON during preprocessing.
- Optimized embeddings for domain-specific retrieval in education.
- Ensured high-precision generation with cleaning and regex extraction.
- Evaluated comprehensively beyond accuracy (e.g., BLEU for explanations).

This project demonstrates end-to-end RAG integration for educational QA, ready for deployment in K-12 scenarios.
