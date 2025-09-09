import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

st.set_page_config(page_title="Chat with your Data", layout="wide")
st.title("📊 Chat with your CSV/Excel using PandasAI")

# Sidebar: Select LLM provider
provider = st.sidebar.selectbox("Choose LLM Provider", ["OpenAI", "OpenRouter"])

# Initialize provider-specific API key and model
api_key = None
api_base = None
model = None

if provider == "OpenAI":
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI key not set. Switch to OpenRouter or provide a key.")
        st.stop()
    api_base = "https://api.openai.com/v1"
    model = st.sidebar.text_input("Model", "gpt-4")

elif provider == "OpenRouter":
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.warning("OpenRouter key not set. Provide a key to use this provider.")
        st.stop()
    api_base = "https://openrouter.ai/api/v1"

    openrouter_models = {
        "openai/gpt-oss-20b:free": "free",
        "openai/gpt-oss-120b:free": "free",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": "free",
        "qwen/qwen3-coder:free": "free"
    }

    model = st.sidebar.selectbox(
        "Model",
        [f"{name} ({label})" for name, label in openrouter_models.items()]
    )
    model = model.split(" ")[0]

else:
    st.error("Unsupported provider")
    st.stop()

# Initialize LLM
llm = OpenAI(api_token=api_key, model=model, api_base=api_base)

# Upload file
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Preview of your data:", df.head())

    # Show column names in sidebar
    st.sidebar.subheader("Data Columns")
    for col in df.columns:
        st.sidebar.text(col)

    smart_df = SmartDataframe(
        df,
        config={
            "llm": llm,
            "enable_cache": False,
            "use_error_correction_framework": True,
            "save_charts": True,
        },
    )

    # Chat input
    query = st.text_area("Ask a question about your data")
    if st.button("Run Query") and query:
        if "history" not in st.session_state:
            st.session_state.history = []

        with st.spinner("Thinking..."):
            response = smart_df.chat(query)

        # Save to chat history
        st.session_state.history.append({"query": query, "response": response})

        # Display all previous Q&A
        for entry in st.session_state.history:
            st.markdown(f"**You:** {entry['query']}")
            if isinstance(entry['response'], str):
                st.markdown(f"**AI:** {entry['response']}")
            else:
                try:
                    st.image(entry['response'], caption="Generated chart")
                except Exception:
                    st.markdown(f"**AI:** {entry['response']}")