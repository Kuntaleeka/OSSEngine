import os
import re
import yaml
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

# Core imports
from memory import ConversationMemory
from llm_client import LLMClient

# Page configuration
st.set_page_config(
    page_title="LLM Benchmarking Suite",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "guardrails.yaml")
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    config = {
        "assistant": {
            "name": "Benchmarking Assistant",
            "system_prompt": "You are a helpful assistant."
        },
        "safety_rules": [],
        "generation_params": {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9
        }
    }

# Some AI help for CSS styling to create a sleek, modern UI with a dark theme and glassmorphism elements
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* Gradient Header */
    .glowing-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        text-shadow: 0 0 40px rgba(99, 102, 241, 0.2);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Glassmorphism Containers & Cards */
    .glass-card {
        background: rgba(17, 25, 40, 0.45);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #cbd5e1;
    }

    .metric-badge-latency {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
    }

    .metric-badge-tokens {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #60a5fa;
    }
    
    .metric-badge-guard {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #f87171;
    }

    /* Chat Styling */
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        max-width: 80%;
        line-height: 1.5;
        position: relative;
    }

    .chat-bubble-user {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }

    .chat-bubble-assistant {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #f1f5f9;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .chat-bubble-refused {
        background: rgba(239, 68, 68, 0.05);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }

    /* Styled Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.3);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)


if "memory" not in st.session_state:
    # Initialize conversational memory
    st.session_state.memory = ConversationMemory(
        system_prompt=config["assistant"]["system_prompt"],
        max_window_size=10
    )

if "client" not in st.session_state:
    st.session_state.client = LLMClient()

if "chat_history" not in st.session_state:
    # Dedicated list of displayable history with metrics
    # List of Dict containing: role, content, latency, tokens_in, tokens_out, provider, model, refused
    st.session_state.chat_history = []

if "performance_log" not in st.session_state:
    st.session_state.performance_log = []

st.sidebar.markdown("<div style='text-align: center; margin-bottom: 1.5rem;'><h2 style='color:#a855f7; font-weight:700; margin-bottom:0;'>⚙️ Settings</h2><span style='color:#64748b;'>Configure Assistants</span></div>", unsafe_allow_html=True)

# Assistant provider switching
provider_type = st.sidebar.radio(
    "Select Model Class",
    ["Open Source Model", "Frontier Model"],
    index=0
)

# Populate models and provider settings
if provider_type == "Open Source Model":
    provider_name = "huggingface"
    default_model = os.getenv("OSS_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    model_input = st.sidebar.text_input("HF Model ID", value=default_model)
else:
    provider_name = "gemini"
    default_model = os.getenv("FRONTIER_MODEL_ID", "gemini-2.5-flash")
    model_input = st.sidebar.text_input("Gemini Model ID", value=default_model)

# Context memory window configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Memory Configuration")
window_size = st.sidebar.slider(
    "Sliding Window Size (turns)",
    min_value=1,
    max_value=20,
    value=10,
    help="Number of recent messages to keep in the context window sent to the LLM."
)
# Update memory window dynamically
st.session_state.memory.max_window_size = window_size

if st.sidebar.button("🗑️ Clear Chat & Memory", use_container_width=True):
    st.session_state.memory.clear()
    st.session_state.chat_history.clear()
    st.sidebar.success("Chat history cleared!")
    st.rerun()

# Guardrail settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Safety & Guardrails")
enable_guardrails = st.sidebar.checkbox("Enable Real-time Guardrails", value=True)

# System Prompt Override
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 System Prompt")
custom_sys_prompt = st.sidebar.text_area(
    "Active System Instructions",
    value=config["assistant"]["system_prompt"],
    height=150
)
# Update system prompt dynamically
st.session_state.memory.system_prompt = custom_sys_prompt

# ----------------- MAIN UI -----------------
st.markdown("<h1 class='glowing-header'>🌌 LLM Benchmarking Suite</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A unified interface for benchmarking Open Source and Frontier Assistant Models</div>", unsafe_allow_html=True)


col_chat, col_obs = st.columns([7, 3])

with col_chat:
    st.markdown("<h3 style='margin-bottom:1rem;'>💬 Assistant Chat</h3>", unsafe_allow_html=True)
    
    # Display chat history from session state
    chat_container = st.container()
    with chat_container:
        for idx, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                user_html = f'<div class="chat-bubble chat-bubble-user">\n\n{chat["content"]}\n\n</div>'
                st.markdown(user_html, unsafe_allow_html=True)
            else:
                # Decide styling
                bubble_class = "chat-bubble-refused" if chat.get("refused", False) else "chat-bubble-assistant"
                refused_badge = '<span class="metric-badge metric-badge-guard">🛡️ Refused by Guardrails</span>' if chat.get("refused", False) else ''
                
                # Render response header as a single line with no newlines to avoid markdown parser confusion
                header_html = (
                    f'<div style="font-weight: 500; font-size:0.85rem; color:#94a3b8; margin-bottom: 0.35rem; display:flex; align-items:center; flex-wrap:wrap;">'
                    f'<span style="color:#a855f7; font-weight:600; margin-right:0.75rem;">{chat["provider"].upper()} ({chat["model"]})</span>'
                    f'{refused_badge}'
                    f'<span class="metric-badge metric-badge-latency">⏱️ {chat["latency"]}s</span>'
                    f'<span class="metric-badge metric-badge-tokens">🔗 In: {chat["tokens_in"]} | Out: {chat["tokens_out"]}</span>'
                    f'</div>'
                )
                

                html_content = (
                    f'<div class="chat-bubble {bubble_class}">'
                    f'{header_html}'
                    f'\n\n'
                    f'{chat["content"]}'
                    f'\n\n'
                    f'</div>'
                )
                st.markdown(html_content, unsafe_allow_html=True)
                
                # Show expander for raw JSON if present in history
                if "raw_json" in chat and chat["raw_json"]:
                    with st.expander("🔍 View Raw JSON Response"):
                        st.json(chat["raw_json"])

    # Chat Input
    prompt = st.chat_input("Ask the assistant something...")
    if prompt:

        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "refused": False
        })
        

        triggered_rule = None
        refusal_msg = ""
        if enable_guardrails:
            for rule in config.get("safety_rules", []):
                pattern = rule.get("pattern", "")
                exception_pattern = rule.get("exception_pattern", "")
                if re.search(pattern, prompt):
                    if exception_pattern and re.search(exception_pattern, prompt):
                        continue
                    triggered_rule = rule.get("id")
                    refusal_msg = rule.get("refusal_message", "I cannot fulfill this request.")
                    break
        
        if triggered_rule:

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": refusal_msg,
                "latency": 0.001,
                "tokens_in": len(prompt) // 4,
                "tokens_out": len(refusal_msg) // 4,
                "provider": "Local Guardrail",
                "model": "regex-filter",
                "refused": True,
                "raw_json": {
                    "text": refusal_msg,
                    "latency": 0.001,
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(refusal_msg) // 4,
                    "provider": "Local Guardrail",
                    "model": "regex-filter",
                    "error": "Refused locally by guardrail rule: " + triggered_rule
                }
            })

            st.session_state.performance_log.append({
                "model_class": provider_type,
                "provider": "local_guardrail",
                "model": "regex-filter",
                "prompt_len": len(prompt),
                "latency": 0.001,
                "refused": True
            })
            st.rerun()
        else:
            st.session_state.memory.add_message("user", prompt)
            
            # Fetch formatted messages 
            history_openai = st.session_state.memory.get_messages_openai()
            history_gemini = st.session_state.memory.get_messages_gemini()
            
            # Fetch settings
            params = config.get("generation_params", {"temperature": 0.7, "max_tokens": 1024, "top_p": 0.9})
            
           
            with st.spinner(f"Querying {provider_type} ({model_input})..."):
                # Call client
                response = st.session_state.client.get_response(
                    provider=provider_name,
                    prompt=prompt,
                    history_messages_openai=history_openai,
                    history_messages_gemini=history_gemini,
                    system_prompt=custom_sys_prompt,
                    model_id=model_input,
                    gen_params=params
                )
            
            # Add assistant message to memory (only if not an API error)
            if not response.get("error"):
                st.session_state.memory.add_message("assistant", response["text"])
            
            # Record in chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["text"],
                "latency": response["latency"],
                "tokens_in": response["prompt_tokens"],
                "tokens_out": response["completion_tokens"],
                "provider": response["provider"],
                "model": response["model"],
                "refused": False,
                "raw_json": response
            })
            
            # Add to performance log
            st.session_state.performance_log.append({
                "model_class": provider_type,
                "provider": response["provider"],
                "model": response["model"],
                "prompt_len": len(prompt),
                "latency": response["latency"],
                "refused": False
            })
            st.rerun()

with col_obs:
    st.markdown("<h3 style='margin-bottom:1rem;'>🔬 Observability & Stats</h3>", unsafe_allow_html=True)
    
    # Render Performance Card
    stats_html = ""
    if st.session_state.performance_log:
        df_perf = pd.DataFrame(st.session_state.performance_log)
        avg_lat = df_perf[df_perf["refused"] == False]["latency"].mean() if not df_perf[df_perf["refused"] == False].empty else 0
        refusals = df_perf["refused"].sum()
        
        stats_html += f"<p style='margin: 0.5rem 0;'><strong>Average Latency:</strong> {avg_lat:.2f} seconds</p>"
        stats_html += f"<p style='margin: 0.5rem 0 1rem 0;'><strong>Guardrail Interventions:</strong> {refusals}</p>"
        
        summary_df = df_perf.groupby("model").agg(
            avg_latency=("latency", "mean"),
            queries=("latency", "count")
        ).reset_index()
        
        table_rows = ""
        for _, row in summary_df.iterrows():
            table_rows += (
                f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">'
                f'<td style="padding: 0.5rem; text-align: left; font-size: 0.85rem;">{row["model"]}</td>'
                f'<td style="padding: 0.5rem; text-align: right; font-size: 0.85rem;">{row["avg_latency"]:.2f}s</td>'
                f'<td style="padding: 0.5rem; text-align: right; font-size: 0.85rem;">{row["queries"]}</td>'
                f'</tr>'
            )
        
        stats_html += (
            f'<table style="width: 100%; border-collapse: collapse; margin-top: 0.5rem;">'
            f'<thead>'
            f'<tr style="border-bottom: 2px solid rgba(255,255,255,0.1); color: #94a3b8; font-size: 0.8rem;">'
            f'<th style="padding: 0.5rem; text-align: left;">Model</th>'
            f'<th style="padding: 0.5rem; text-align: right;">Avg Lat</th>'
            f'<th style="padding: 0.5rem; text-align: right;">Count</th>'
            f'</tr>'
            f'</thead>'
            f'<tbody>'
            f'{table_rows}'
            f'</tbody>'
            f'</table>'
        )
    else:
        stats_html += "<div style='padding: 0.8rem; background: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.15); border-radius: 8px; color: #93c5fd; font-size: 0.85rem;'>No queries sent yet. Performance stats will populate here.</div>"
        
    st.markdown(
        f'<div class="glass-card">'
        f'<h4 style="margin-top:0; color:#3b82f6;">📈 Session Analytics</h4>'
        f'{stats_html}'
        f'</div>',
        unsafe_allow_html=True
    )
        
    # Render Memory Context Inspector
    active_window = st.session_state.memory.get_windowed_raw_history()
    memory_html = f"<p style='margin: 0.5rem 0;'><strong>Active window size:</strong> <code>{window_size}</code> messages.</p>"
    
    if active_window:
        memory_html += "<p style='font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.75rem;'>Active context history currently passed to the LLM:</p>"
        for m in active_window:
            role_label = "👤 User" if m["role"] == "user" else "🤖 Assistant"
            role_color = "#3b82f6" if m["role"] == "user" else "#a855f7"
            snippet = m['content'][:80].replace("<", "&lt;").replace(">", "&gt;") + ("..." if len(m['content']) > 80 else "")
            memory_html += (
                f'<div style="margin-bottom: 0.5rem; padding: 0.4rem 0.6rem; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 6px; font-size: 0.85rem;">'
                f'<span style="color: {role_color}; font-weight: 600;">{role_label}:</span> {snippet}'
                f'</div>'
            )
        
        import json
        raw_context_data = {
            "system_prompt": custom_sys_prompt,
            "windowed_history": active_window
        }
        json_str = json.dumps(raw_context_data, indent=2)
        
        memory_html += (
            f'<details style="margin-top: 0.75rem; font-size: 0.85rem;">'
            f'<summary style="cursor: pointer; color: #a855f7; font-weight: 500; outline: none; margin-bottom: 0.5rem;">🔍 View Raw Context JSON</summary>'
            f'<pre style="background: rgba(0, 0, 0, 0.3); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); overflow-x: auto; color: #cbd5e1; font-family: monospace; font-size: 0.8rem; white-space: pre-wrap; max-height: 250px;">{json_str}</pre>'
            f'</details>'
        )
    else:
        memory_html += "<div style='padding: 0.8rem; background: rgba(168, 85, 247, 0.05); border: 1px solid rgba(168, 85, 247, 0.15); border-radius: 8px; color: #d8b4fe; font-size: 0.85rem;'>Context memory is currently empty. Start chatting to inspect context.</div>"
        
    st.markdown(
        f'<div class="glass-card">'
        f'<h4 style="margin-top:0; color:#a855f7;">🧠 Memory Inspector</h4>'
        f'{memory_html}'
        f'</div>',
        unsafe_allow_html=True
    )

    # Render API Key Status
    status_gemini = "🟢 Configured" if st.session_state.client.gemini_api_key else "🔴 Missing"
    status_hf = "🟢 Configured" if st.session_state.client.hf_token else "⚪ Missing (Public Rate Limits apply)"
    
    credentials_html = (
        f'<div style="font-size: 0.9rem; line-height: 1.6;">'
        f'<p style="margin: 0.4rem 0;"><strong>Gemini API Key:</strong> {status_gemini}</p>'
        f'<p style="margin: 0.4rem 0;"><strong>HF Token:</strong> {status_hf}</p>'
        f'<p style="margin: 1rem 0 0 0; font-size: 0.8rem; color: #64748b; line-height: 1.4;">'
        f'Setup your keys in the <code>.env</code> file at the root of the project folder.'
        f'</p>'
        f'</div>'
    )
    
    st.markdown(
        f'<div class="glass-card">'
        f'<h4 style="margin-top:0; color:#10b981;">🔌 Credentials & Status</h4>'
        f'{credentials_html}'
        f'</div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; margin-bottom: 1.5rem; color: #64748b; font-size: 0.85rem;">
    made by K. Kundu
</div>
""", unsafe_allow_html=True)
