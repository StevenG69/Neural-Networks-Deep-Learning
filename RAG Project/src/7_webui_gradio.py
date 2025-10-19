# src/7_webui_gradio.py

import logging
import gradio as gr
from pathlib import Path
from rag_chain import load_vector_db, load_model_and_tokenizer, retrieve_context
from rag_chain import build_cot_prompt_for_inference, generate_answer, extract_answer, parse_choices

BASE_DIR = Path(__file__).parent
FINETUNED_EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "finetuned_embedding"
FINETUNED_MODEL_PATH = BASE_DIR / "models" / "finetuned_model"
VECTOR_DB_PATH = BASE_DIR / "data" / "processed" / "vector_db"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
model, tokenizer, db = None, None, None

def initialize_rag_components():
    global model, tokenizer, db
    if all([model, tokenizer, db]): return True
    try:
        db = load_vector_db(VECTOR_DB_PATH, FINETUNED_EMBEDDING_MODEL_PATH)
        model, tokenizer = load_model_and_tokenizer(FINETUNED_MODEL_PATH)
        return True
    except Exception as e:
        logger.error(f"Init failed: {e}")
        return False

def predict_answer(question, choices_str, topic=None):
    global model, tokenizer, db
    if not initialize_rag_components() or not question.strip():
        return "Please enter a valid question.", "", ""
    choices = parse_choices(choices_str)
    if not choices: return "Invalid choices format.", "", ""
    context = retrieve_context(db, question, k=2)
    response = generate_answer(model, tokenizer, build_cot_prompt_for_inference(question, choices, context))
    result = extract_answer(response, "", len(choices))
    return (
        f"**Index:** {result['index']}\n**Choice:** {choices[result['index']]}\n**Explanation:** {result['explanation']}" 
        if result['index'] is not None and 0 <= result['index'] < len(choices)
        else f"**Error:** Invalid index.\n**Output:** {response}",
        context,
        response
    )

def create_ui():
    with gr.Blocks(title="ScienceQA Chatbot (High Precision RAG)", 
                   css=".footer {position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #f0f0f0;}") as demo:
        gr.Markdown("# ScienceQA Chatbot")
        gr.Markdown("Ask a multiple-choice science question, provide the choices, and optionally a topic. The chatbot will use retrieval and reasoning to answer.")
        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    value="Enter your science question...", 
                    lines=3, 
                    label="Question",
                    interactive=True  
                )
                choices_input = gr.Textbox(
                    value="'option1', 'option2', ...", 
                    lines=2, 
                    label="Choices",
                    interactive=True  
                )
                topic_input = gr.Textbox(
                    value="e.g., biology", 
                    label="Topic (Optional)",
                    interactive=True  
                )
                submit_btn = gr.Button("Get Answer", variant="primary")
            
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")
                with gr.Accordion("Retrieved Context", open=False):
                    context_output = gr.Textbox(
                        lines=10, 
                        interactive=False
                    )
                with gr.Accordion("Full Output", open=False):
                    full_output = gr.Textbox(
                        lines=10, 
                        interactive=False
                    )
        gr.Markdown('<div class="footer">Science QA | Chatbot</div>')
        submit_btn.click(
            predict_answer, 
            inputs=[question_input, choices_input, topic_input],
            outputs=[answer_output, context_output, full_output]
        )
    return demo

if __name__ == "__main__":
    logger.info(f"Starting UI | Embedding: {FINETUNED_EMBEDDING_MODEL_PATH} | Model: {FINETUNED_MODEL_PATH}")
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_api=False
    )