# 🧠 GPT-From-Scratch :

A custom-built, **355-million parameter Decoder-Only Transformer** (GPT-2 Medium architecture) implemented entirely from scratch in PyTorch, loaded with pre-trained weights, and instruction fine-tuned using Low-Rank Adaptation (LoRA).

This project features a hybrid cloud/local pipeline: train efficiently on free cloud GPUs (Colab/Kaggle) and interact with your model locally using a beautifully styled **Streamlit Dashboard**.

---

## 🏗️ Project Architecture & Features

This project serves as an end-to-end demonstration of the complete LLM lifecycle:

* **From-Scratch Transformer (`src/model.py`):** Custom implementations of Causal Multi-Head Attention (with triangular masking), Layer Normalization, FeedForward layers, and GELU activations. No high-level wrappers (like Hugging Face `transformers` modules) are used in the core model definition.
* **Weight Tying Optimization:** Shares parameter weights between the input token embedding layer and the final output head, reducing memory consumption by ~30%.
* **LoRA (Low-Rank Adaptation) (`src/lora.py`):** Freezes base pre-trained weights and injects trainable rank decomposition matrices ($A$ and $B$, rank=16) into the linear layers, reducing trainable parameters by >95% for efficient SFT.
* **Advanced Training Setup (`src/training.py`):** Stable training loop utilizing **Learning Rate Warmup**, **Cosine Decay scheduling**, and **Gradient Clipping**.
* **Smart Data Loader (`src/dataset.py`):** Implements dynamic padding to optimize sequence processing and **Target Masking** (replacing instruction prompts and padding with `-100` targets) so the model only calculates loss on generated responses.

---

## 📁 Repository Structure

```text
LLM_from_Scratch/
├── src/
│   ├── model.py          # Custom GPT architecture & weight tying
│   ├── lora.py           # LoRALayer and dynamic replacement utility
│   ├── dataset.py        # Instruction dataset formatter and target masking
│   ├── training.py       # Warmup, Cosine decay, and training loops
│   └── weights.py        # OpenAI GPT-2 weight downloader and mapper
├── notebooks/
│   └── cloud_training.ipynb  # Notebook for Colab/Kaggle fine-tuning
├── app.py                # Local Streamlit web application
├── requirements.txt      # Python dependencies
└── data/                 # Training datasets
    ├── the-verdict.txt
    └── instruction-data.json
```

---

## 🚀 How to Run It

### Step 1: Cloud Fine-Tuning (Colab/Kaggle)
1. Upload the `LLM_from_Scratch/` directory to Google Colab or Kaggle.
2. Ensure you have a **T4 GPU** enabled in your environment.
3. Open `notebooks/cloud_training.ipynb` and run all cells.
4. After 2 epochs (~5 minutes), download the generated **`lora_weights.pth`** file (~10MB) to your local computer and place it in the root folder of this project.

### Step 2: Run the Local App
Launch the Streamlit dashboard on your local computer to chat with your model:

```bash
# Navigate to the project directory
cd LLM_from_Scratch

# Install dependencies
pip install -r requirements.txt

# Run the web app
python3 -m streamlit run app.py
```

---

## 💬 The User Experience (Streamlit Tabs)

1. **Tab 1: Text Generation Sandbox:** Prompt the raw pre-trained model. Watch it act as a next-token autocompleter (e.g., generating repeating patterns instead of answering questions).
2. **Tab 2: Niche Chatbot:** Swap to your `lora_weights.pth` file in the sidebar and chat with the model. Experience the "Aha!" moment of SFT alignment as the model starts answering questions.
3. **Tab 3: Zero-Shot Tasks:** Test the model's emergent translation and summarization abilities using template formatting.
