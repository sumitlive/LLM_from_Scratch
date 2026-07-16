import os
import streamlit as st
import torch
import tiktoken
import time

from src.model import GPTModel
from src.weights import download_and_load_gpt2
from src.lora import replace_linear_with_lora
from src.training import generate, text_to_token_ids, token_ids_to_text

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="LLM Sandbox", page_icon="🧠", layout="wide")

st.title("🧠 The LLM Sandbox")
st.markdown("""
*A custom-built 355M parameter Large Language Model, built entirely from scratch in PyTorch.*
Explore the tabs to see how a model evolves from a simple next-word predictor (Base Model) to a helpful instruction-following assistant (Fine-Tuned Model).
""")

# ==========================================
# 2. CACHED MODEL LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_base_model():
    # Instantiate custom architecture
    GPT_CONFIG_355M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_355M)
    model.eval()
    
    # Try to load base weights from HF
    try:
        download_and_load_gpt2(model, "355M")
    except Exception as e:
        st.error(f"Failed to load GPT-2 Base weights. Please check your internet connection or transformers installation.\n{e}")
        
    model.to(device)
    return model, device

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Model Configuration")

# Check if lora_weights.pth exists
lora_exists = os.path.exists("lora_weights.pth")

weight_mode = st.sidebar.radio(
    "Select Model Weights:",
    ("Pre-trained (Base GPT-2)", "Fine-Tuned (Instruction + LoRA)"),
    index=1 if lora_exists else 0,
    help="Toggle between the raw base model and your LoRA fine-tuned model."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1, 
                                help="Controls creativity. 0.0 is completely deterministic.")
top_k = st.sidebar.slider("Top-k", min_value=1, max_value=100, value=25, step=1, 
                          help="Limits the model to predicting only from the top 'k' most likely next words.")
max_tokens = st.sidebar.slider("Max New Tokens", min_value=10, max_value=250, value=50, step=10)

# ==========================================
# 4. INITIALIZATION & WEIGHT TOGGLING
# ==========================================
with st.spinner("Loading base model architecture..."):
    base_model, device = load_base_model()
    tokenizer = get_tokenizer()

def apply_lora_if_needed(model, weight_mode):
    if weight_mode == "Fine-Tuned (Instruction + LoRA)":
        if not os.path.exists("lora_weights.pth"):
            st.sidebar.error("⚠️ lora_weights.pth not found! Please run the cloud training notebook first, or stick to Pre-trained mode.")
            return model, False
            
        # We need to ensure LoRA layers exist. Since we cache the base model, 
        # checking if the first block's query layer is LinearWithLoRA is a good proxy.
        from src.lora import LinearWithLoRA, replace_linear_with_lora
        if not isinstance(model.trf_blocks[0].att.W_query, LinearWithLoRA):
            replace_linear_with_lora(model, rank=16, alpha=16)
            
        # Load the LoRA weights
        try:
            model.load_state_dict(torch.load("lora_weights.pth", map_location=device), strict=False)
            model.to(device)
            return model, True
        except Exception as e:
            st.sidebar.error(f"Error loading LoRA weights: {e}")
            return model, False
    else:
        # If user selects Base Model but LoRA is injected, we'd theoretically need to un-inject it.
        # For simplicity in this demo, if they switch back, we just set alpha=0 or reload.
        # A quick hack is to temporarily set LoRA alpha to 0 for inference.
        from src.lora import LinearWithLoRA
        if isinstance(base_model.trf_blocks[0].att.W_query, LinearWithLoRA):
            for name, module in base_model.named_modules():
                if isinstance(module, LinearWithLoRA):
                    module.lora.alpha = 0  # Disable LoRA influence
        return base_model, True

model, lora_loaded = apply_lora_if_needed(base_model, weight_mode)
if weight_mode == "Fine-Tuned (Instruction + LoRA)" and lora_loaded:
    from src.lora import LinearWithLoRA
    # Restore alpha to 16 if we previously zeroed it out
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            module.lora.alpha = 16

# Helper for generation
def generate_response(prompt_text):
    encoded = text_to_token_ids(prompt_text, tokenizer).to(device)
    context_size = model.pos_emb.weight.shape[0]
    
    with st.spinner("Generating..."):
        start_time = time.time()
        with torch.no_grad():
            out_ids = generate(
                model=model, 
                idx=encoded, 
                max_new_tokens=max_tokens, 
                context_size=context_size,
                temperature=temperature,
                top_k=top_k,
                eos_id=tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
            )
        elapsed = time.time() - start_time
        
    generated_text = token_ids_to_text(out_ids, tokenizer)
    return generated_text, elapsed

# ==========================================
# 5. TABS UI
# ==========================================
tab1, tab2, tab3 = st.tabs(["💬 1. Text Generation Sandbox", "🤖 2. Niche Chatbot", "✨ 3. Zero-Shot Tasks"])

# --- TAB 1: Base Text Generation ---
with tab1:
    st.header("Text Generation Sandbox")
    st.markdown("Test how the model acts as an advanced autocomplete engine. *Best used with the Base Model.*")
    
    base_prompt = st.text_area("Enter a prompt:", "The greatest invention in history is", height=100)
    if st.button("Generate Text"):
        result, t = generate_response(base_prompt)
        st.success(f"Generated in {t:.2f} seconds.")
        st.text_area("Output:", result, height=250)

# --- TAB 2: Instruction Chatbot ---
with tab2:
    st.header("Niche Chatbot")
    st.markdown("Interact with the model as an assistant. *Best used with the Fine-Tuned Model.*")
    
    chat_instruction = st.text_area("User Request:", "Write a polite email declining a job offer.", height=100)
    if st.button("Send Request"):
        # Format as Alpaca Prompt
        alpaca_prompt = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{chat_instruction}"
            f"\n\n### Response:\n"
        )
        
        result, t = generate_response(alpaca_prompt)
        
        # Parse out just the response
        try:
            response_only = result.split("### Response:\n")[1]
        except IndexError:
            response_only = result
            
        st.success(f"Generated in {t:.2f} seconds.")
        st.info(response_only)

# --- TAB 3: Zero-Shot Tasks ---
with tab3:
    st.header("Zero-Shot Translation & Summarization")
    st.markdown("Demonstrate emergent abilities by using clever prompt formatting.")
    
    task_type = st.selectbox("Select Task:", ["Summarization", "Translation (English to French)"])
    
    if task_type == "Summarization":
        source_text = st.text_area("Text to summarize:", "Large language models are artificial neural networks designed to understand and generate human language. They are trained on massive amounts of text data, allowing them to capture complex patterns of grammar, facts, and reasoning abilities.", height=150)
        task_prompt = f"Summarize the following text briefly:\n\n{source_text}\n\nSummary:\n"
    else:
        source_text = st.text_area("Text to translate:", "Hello, where is the nearest train station?", height=100)
        task_prompt = f"Translate English to French:\n\nEnglish: {source_text}\nFrench:"
        
    if st.button("Execute Task"):
        result, t = generate_response(task_prompt)
        
        try:
            if task_type == "Summarization":
                output_only = result.split("Summary:\n")[1]
            else:
                output_only = result.split("French:")[1]
        except IndexError:
            output_only = result
            
        st.success(f"Generated in {t:.2f} seconds.")
        st.warning(output_only)
