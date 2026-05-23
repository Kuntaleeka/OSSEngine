# 🌌 LLM Benchmarking Suite: AI Personal Assistant Benchmarking

A unified framework to run, compare, and benchmark Open Source (OSS) models (e.g. Qwen 2.5 via Hugging Face Serverless Inference) against Frontier models (e.g. Google Gemini). 

This project features:
1. **Multi-turn conversation history** with a configurable sliding window.
2. **Front-line guardrails** configured in YAML with regex patterns to intercept harmful prompts locally.
3. **Observability dashboards** built directly into the UI showing latency, token usage, and guardrail flags.
4. **Automated LLM-as-a-judge benchmarking** framework to score models on Factual Accuracy (Hallucination), Bias & Harm, and Content Safety.

---

## 📁 Repository Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py            # Streamlit interface with premium styling
│   ├── llm_client.py      # LLM client wrapping OSS & Frontier
│   └── memory.py          # Chat memory (sliding window/buffer)
├── configs/
│   └── guardrails.yaml    # System prompts & safety settings
├── evals/
│   ├── dataset.json       # Benchmarking dataset of factual, adversarial, and bias prompts
│   └── judge.py           # LLM-as-a-judge evaluation harness
├── .env.example           # Template for environment variables
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## ⚡ Setup & Local Execution

### 1. Clone the repository and navigate to the project directory
```bash
git clone <repo-url>
cd assistant-project
```

### 2. Create and activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```

Ensure `.env` contains:
```env
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_hugging_face_token_here
```

### 5. Launch the Streamlit Interface
```bash
streamlit run app/main.py
```

### 6. Run the Automated Evaluation Harness
To benchmark the models on the prompt dataset and run the LLM-as-a-judge:
```bash
python evals/judge.py
```
This will print a summary of metrics and output the detailed evaluation results to `evals/results.json`.

---

## 🛡️ Architecture & Design Decisions

- **Sliding Window Memory**: Avoids context blow-up and excessive token costs by capping context to the last $N$ turns (adjustable via UI). The memory is formatted dynamically to match the specific model API schemas (e.g. converting `assistant` roles to `model` for Gemini).
- **Dual-Layer Guardrails**: Combines lightweight, ultra-fast regex guardrails directly at the application boundary for common risk factors, with standard system prompt constraints for nuanced safe generation.
- **Provider-Agnostic LLM Interface**: A single class `LLMClient` wraps all providers (Gemini and Hugging Face). This decouples application logic from specific vendor SDK quirks.
- **Modern UI Aesthetic**: Built with a dark mode glassmorphic UI using customized CSS, tailored color palettes, Outfit typography, and dynamic metric badges (latency & tokens) for a premium look and feel.
