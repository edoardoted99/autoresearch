"""
AutoResearch Dashboard — Streamlit app for monitoring autonomous GPT training experiments.

Connects to the remote Mac via SSH to fetch logs, results, and run inference.
Includes Ollama-powered agent for discussing experiments.

Usage: streamlit run dashboard.py
"""

import os
import io
import json
import subprocess

import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="AutoResearch Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SSH_PASSWORD = os.environ.get("SSH_PASSWORD", "qazxcvbnm")
SSH_PORT = os.environ.get("SSH_PORT", "26996")
SSH_USER_HOST = os.environ.get("SSH_USER_HOST", "edoardo.tedesco@openport.io")
REMOTE_DIR = os.environ.get("REMOTE_DIR", "~/coding/autoresearch")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11435")

SSH_BASE = [
    "sshpass", "-p", SSH_PASSWORD,
    "ssh", "-p", SSH_PORT,
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    SSH_USER_HOST,
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ssh_command(cmd: str, timeout: int = 30) -> str:
    """Run a command on the remote Mac via SSH and return stdout."""
    try:
        result = subprocess.run(
            SSH_BASE + [cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0 and result.stderr:
            return f"[ERROR] {result.stderr.strip()}"
        return result.stdout
    except subprocess.TimeoutExpired:
        return "[ERROR] SSH command timed out"
    except FileNotFoundError:
        return "[ERROR] sshpass not found — install it with: sudo apt install sshpass"
    except Exception as e:
        return f"[ERROR] {e}"


def parse_results_tsv(content: str) -> pd.DataFrame:
    if not content or content.startswith("[ERROR]"):
        return pd.DataFrame(columns=["commit", "val_bpb", "memory_gb", "status", "description"])
    try:
        df = pd.read_csv(io.StringIO(content), sep="\t")
        df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
        df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
        df["status"] = df["status"].astype(str).str.strip().str.upper()
        return df
    except Exception:
        return pd.DataFrame(columns=["commit", "val_bpb", "memory_gb", "status", "description"])


@st.cache_data(ttl=30)
def fetch_results():
    return ssh_command(f"cat {REMOTE_DIR}/results.tsv")


@st.cache_data(ttl=60)
def list_checkpoints():
    output = ssh_command(
        f"find {REMOTE_DIR} {REMOTE_DIR}/checkpoints -maxdepth 1 -name '*.pt' 2>/dev/null",
        timeout=10,
    )
    if output.startswith("[ERROR]"):
        return []
    return [f.strip() for f in output.strip().splitlines() if f.strip()]


@st.cache_data(ttl=30)
def list_ollama_models():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def ollama_chat_stream(model, messages):
    """Stream chat completion from Ollama. Yields text chunks."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
    except Exception as e:
        yield f"\n\n[ERROR] {e}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("AutoResearch")
st.sidebar.caption("Autonomous GPT training monitor")
st.sidebar.divider()

st.sidebar.subheader("GPT Chat settings")
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.8, 0.05, key="temperature")
max_tokens = st.sidebar.slider("Max tokens", 10, 500, 200, 10, key="max_tokens")

st.sidebar.divider()
st.sidebar.subheader("Agent settings")
ollama_models = list_ollama_models()
agent_model = st.sidebar.selectbox(
    "Ollama model",
    ollama_models if ollama_models else ["(no models found)"],
    key="agent_model",
)

st.sidebar.divider()
st.sidebar.caption(f"Remote: {SSH_USER_HOST}:{SSH_PORT}")
st.sidebar.caption(f"Ollama: {OLLAMA_URL}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_monitor, tab_compare, tab_chat, tab_agent = st.tabs(
    ["Monitor", "Compare", "Chat (GPT)", "Agent"]
)

# ========================== TAB 1: MONITOR ==========================

with tab_monitor:
    st.header("Experiment Monitor")

    col_log, col_git = st.columns([2, 1])

    with col_log:
        st.subheader("Run Log (last 20 lines)")
        if st.button("Refresh", key="refresh_log"):
            st.cache_data.clear()
        log_output = ssh_command(f"tail -n 20 {REMOTE_DIR}/run.log")
        st.code(log_output, language="text")

    with col_git:
        st.subheader("Git Log (Mac)")
        git_log = ssh_command(f"cd {REMOTE_DIR} && git log --oneline -10")
        st.code(git_log, language="text")

    st.divider()

    st.subheader("Experiment Results")
    results_raw = fetch_results()
    df_results = parse_results_tsv(results_raw)

    if df_results.empty:
        st.info("No results found — results.tsv may not exist yet on the Mac.")
    else:
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        df_ok = df_results[df_results["status"] != "CRASH"].copy()
        df_ok = df_ok.dropna(subset=["val_bpb"])
        if not df_ok.empty:
            st.subheader("Validation BPB over Experiments")
            df_ok = df_ok.reset_index(drop=True)
            df_ok["experiment"] = df_ok.index + 1
            st.line_chart(df_ok, x="experiment", y="val_bpb")

# ========================== TAB 2: COMPARE ==========================

with tab_compare:
    st.header("Compare Experiments")

    results_raw_cmp = fetch_results()
    df_cmp = parse_results_tsv(results_raw_cmp)

    if df_cmp.empty or len(df_cmp) < 2:
        st.info("Need at least 2 experiments in results.tsv to compare.")
    else:
        labels = []
        for i, row in df_cmp.iterrows():
            short_commit = str(row["commit"])[:8]
            desc = str(row.get("description", ""))[:50]
            labels.append(f"#{i + 1}  {short_commit}  —  {desc}")

        col_a, col_b = st.columns(2)
        with col_a:
            sel_a = st.selectbox("Experiment A", labels, index=0, key="cmp_a")
        with col_b:
            sel_b = st.selectbox("Experiment B", labels, index=min(1, len(labels) - 1), key="cmp_b")

        idx_a = labels.index(sel_a)
        idx_b = labels.index(sel_b)

        if idx_a == idx_b:
            st.warning("Select two different experiments to compare.")
        else:
            row_a = df_cmp.iloc[idx_a]
            row_b = df_cmp.iloc[idx_b]

            st.subheader("Metrics Comparison")
            metric_cols = st.columns(2)
            for col, row, label in zip(metric_cols, [row_a, row_b], [sel_a, sel_b]):
                with col:
                    st.markdown(f"**{label}**")
                    st.metric("val_bpb", f"{row['val_bpb']:.6f}" if pd.notna(row["val_bpb"]) else "N/A")
                    st.metric("memory_gb", f"{row['memory_gb']:.1f}" if pd.notna(row["memory_gb"]) else "N/A")
                    st.metric("status", row["status"])
                    st.text(f"commit: {row['commit']}")
                    st.text(f"description: {row.get('description', '')}")

            if pd.notna(row_a["val_bpb"]) and pd.notna(row_b["val_bpb"]):
                delta = row_b["val_bpb"] - row_a["val_bpb"]
                direction = "lower (better)" if delta < 0 else "higher (worse)" if delta > 0 else "same"
                st.info(f"BPB delta (B - A): {delta:+.6f} — B is {direction}")

            st.subheader("Git Diff (train.py)")
            commit_a = str(row_a["commit"]).strip()
            commit_b = str(row_b["commit"]).strip()
            diff_output = ssh_command(
                f"cd {REMOTE_DIR} && git diff {commit_a} {commit_b} -- train.py",
                timeout=15,
            )
            if diff_output.strip():
                st.code(diff_output, language="diff")
            else:
                st.info("No diff found (commits may be identical or not found).")

# ========================== TAB 3: CHAT (GPT) ==========================

with tab_chat:
    st.header("Chat with Trained GPT")

    checkpoints = list_checkpoints()
    if checkpoints:
        selected_ckpt = st.selectbox("Checkpoint", checkpoints, key="ckpt_select")
    else:
        selected_ckpt = None
        st.warning("No .pt checkpoint files found on the Mac. Run train.py first.")

    # Chat container for messages
    chat_container = st.container(height=500)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Controls row
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.session_state.chat_messages and st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

    # Chat input (always at bottom)
    if prompt := st.chat_input("Send a prompt to the trained GPT...", key="gpt_chat_input"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        if selected_ckpt is None:
            response = "No checkpoint selected."
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        else:
            safe_prompt = prompt.replace("'", "'\\''")
            gen_cmd = (
                f"cd {REMOTE_DIR} && ~/.local/bin/uv run generate.py"
                f" --checkpoint '{selected_ckpt}'"
                f" --prompt '{safe_prompt}'"
                f" --tokens {max_tokens}"
                f" --temperature {temperature}"
            )
            response = ssh_command(gen_cmd, timeout=120)
            if "---" in response:
                response = response.split("---", 1)[1].strip()
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

        st.rerun()

# ========================== TAB 4: AGENT ==========================

with tab_agent:
    st.header("Research Agent")
    st.caption("Chat with an LLM via Ollama to discuss experiments, get ideas, analyze results.")

    if not ollama_models or agent_model == "(no models found)":
        st.error(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            "Make sure the SSH tunnel is running:\n\n"
            "`ssh -p 26996 -L 0.0.0.0:11435:localhost:11434 edoardo.tedesco@openport.io`"
        )
    else:
        # Build system prompt with current experiment context
        @st.cache_data(ttl=30)
        def get_experiment_context():
            results = fetch_results()
            train_py = ssh_command(f"head -50 {REMOTE_DIR}/train.py")
            run_log = ssh_command(f"tail -10 {REMOTE_DIR}/run.log")
            return results, train_py, run_log

        results_ctx, train_ctx, log_ctx = get_experiment_context()

        system_prompt = f"""You are a helpful AI research assistant for the AutoResearch project.
This project autonomously trains small GPT models and experiments with architecture/hyperparameters.
The metric is val_bpb (validation bits per byte) — lower is better.

Current experiment results (TSV):
{results_ctx}

Current train.py hyperparameters (first 50 lines):
{train_ctx}

Latest training log:
{log_ctx}

Help the user analyze results, suggest experiments, debug issues, and discuss ML concepts.
Be concise and practical."""

        # Agent chat container
        agent_container = st.container(height=500)

        if "agent_messages" not in st.session_state:
            st.session_state.agent_messages = []

        with agent_container:
            for msg in st.session_state.agent_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Controls
        col_clear_agent, _ = st.columns([1, 5])
        with col_clear_agent:
            if st.session_state.agent_messages and st.button("Clear chat", key="clear_agent"):
                st.session_state.agent_messages = []
                st.rerun()

        # Agent chat input
        if agent_prompt := st.chat_input("Ask the research agent...", key="agent_chat_input"):
            st.session_state.agent_messages.append({"role": "user", "content": agent_prompt})

            # Build messages for Ollama (system + history)
            ollama_messages = [{"role": "system", "content": system_prompt}]
            ollama_messages.extend(st.session_state.agent_messages)

            # Stream response
            with agent_container:
                with st.chat_message("user"):
                    st.markdown(agent_prompt)
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        ollama_chat_stream(agent_model, ollama_messages)
                    )

            st.session_state.agent_messages.append({"role": "assistant", "content": response})
