# LLM Benchmarking Suite: AI Personal Assistant Benchmarking

A unified framework to run, compare, and benchmark Open Source (OSS) models (e.g. Qwen 2.5 via Hugging Face Serverless Inference) against Frontier models (e.g. Google Gemini). 

This project features:
1. **Multi-turn conversation history** with a configurable sliding window.
2. **Front-line guardrails** configured in YAML with regex patterns to intercept harmful prompts locally.
3. **Observability dashboards** built directly into the UI showing latency, token usage, and guardrail flags.
4. **Automated LLM-as-a-judge benchmarking** framework to score models on Factual Accuracy (Hallucination), Bias & Harm, and Content Safety.

---

## Repository Structure

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

## Architecture & Design Decisions

- **Sliding Window Memory**: Avoids context blow-up and excessive token costs by capping context to the last $N$ turns (adjustable via UI). The memory is formatted dynamically to match the specific model API schemas (e.g. converting `assistant` roles to `model` for Gemini).
- **Dual-Layer Guardrails**: Combines lightweight, ultra-fast regex guardrails directly at the application boundary for common risk factors, with standard system prompt constraints for nuanced safe generation.
- **Provider-Agnostic LLM Interface**: A single class `LLMClient` wraps all providers (Gemini and Hugging Face). This decouples application logic from specific vendor SDK quirks.
- **Modern UI Aesthetic**: Built with a dark mode glassmorphic UI using customized CSS, tailored color palettes, Outfit typography, and dynamic metric badges (latency & tokens) for a premium look and feel.

---

##  Tradeoffs Made

- **Local Regex Guardrails vs. LLM-Based Moderation**: 
  - *Tradeoff*: We chose localized regex patterns over an online guardrail model (e.g., Llama Guard).
  - *Rationale*: Regex runs in sub-milliseconds with zero token overhead and cost. To overcome word-rigidity, we implemented `exception_pattern` overrides (e.g., allowing "defuse a bomb" but blocking "build a bomb").
- **Hugging Face Serverless API vs. Dedicated Endpoints**:
  - *Tradeoff*: We chose the Hugging Face Serverless API for the OSS model instead of deploying a dedicated inference server.
  - *Rationale*: The serverless API is completely free and requires no backend infrastructure, making it ideal for testing. However, it is subject to rate-limiting queues and latency fluctuations compared to dedicated hosted endpoints.
- **Sliding Window Memory vs. Summarization Memory**:
  - *Tradeoff*: We used a fixed sliding window of size $N$ rather than dynamic history summarization.
  - *Rationale*: Keeping a sliding window of recent messages ensures query context remains predictable and avoids the latency and API cost of a secondary "summarization" call on every message.

---

##  Future Improvements (With More Time)

- **Semantic Local Guardrails**: Replace keyword regular expressions with a local, lightweight vector embedding model (e.g. ONNX-run SentenceTransformers) to measure prompt safety using cosine similarity.
- **Asynchronous Benchmarking Engine**: Refactor the evaluation runner (`evals/judge.py`) using `asyncio` to speed up tests, and implement automatic mocking fallbacks when api rate limits are encountered.
- **Memory Context Summarization**: Implement dynamic token tracking that compresses conversation logs into a concise summary once the token count nears the model context limit, instead of strictly discarding old turns.
- **Observability Tracing Integration**: Plug in semantic observability suites (like Phoenix, LangSmith, or Weights & Biases) to capture detailed execution traces, latency histograms, and cost projection reports.
