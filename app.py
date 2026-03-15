import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import get_settings
from models.llm import get_chat_model, get_available_providers
from models.embeddings import get_embedding_model
from utils.prompting import build_system_prompt
from utils.rag import load_documents_from_directory, build_retriever, retrieve_context
from utils.web_search import should_trigger_web_search, search_web, format_web_results
import os
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the chat model"""
    try:
        # Message preparation for the model
        formatted_messages = [SystemMessage(content=system_prompt)]
        
        # conversation history added
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        # getting response from the model
        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        return f"Error getting response: {str(e)}"


@st.cache_resource(show_spinner=False)
def initialize_retriever():
    try:
        settings = get_settings()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        kb_dir = os.path.join(base_dir, settings.knowledge_base_dir)
        documents, warnings = load_documents_from_directory(kb_dir)
        if warnings:
            st.session_state["rag_warnings"] = warnings
        if not documents:
            return None
        embedding_model = get_embedding_model()
        return build_retriever(documents, embedding_model)
    except Exception as error:
        st.session_state["rag_warnings"] = [f"RAG initialization failed: {error}"]
        return None

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ##  Installation
                
    
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    
    You'll need API keys from your chosen provider. Get them from:
    
    ### OpenAI
    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    - Create a new API key
    - Set the variables in config
    
    ### Groq
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key
    - Set the variables in config
    
    ### Google Gemini
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Create a new API key
    - Set the variables in config
    
    ## Available Models
    
    ### OpenAI Models
    Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
    Popular models include:
    - `gpt-4o` - Latest GPT-4 Omni model
    - `gpt-4o-mini` - Faster, cost-effective version
    - `gpt-3.5-turbo` - Fast and affordable
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.1-70b-versatile` - Large, powerful model
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    - `gemini-1.5-flash` - Fast and efficient
    - `gemini-pro` - Standard model
    
    ## How to Use
    
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!
    
    ## Tips
    
    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities and costs
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh
    
    ## Troubleshooting
    
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status
    
    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)

def chat_page():
    """Main chat interface page"""
    st.title("NeoStats Insight Assistant")

    settings = get_settings()

    with st.sidebar:
        st.subheader("Assistant Settings")
        provider_options = get_available_providers()
        if not provider_options:
            provider_options = ["groq", "openai", "gemini"]
        selected_provider = st.selectbox("LLM Provider", options=provider_options, index=0)
        response_mode = st.radio("Response Mode", ["Concise", "Detailed"], index=0)
        enable_rag = st.toggle("Use local knowledge base (RAG)", value=True)
        enable_web_search = st.toggle("Enable live web search", value=True)

    try:
        chat_model = get_chat_model(selected_provider)
    except Exception as error:
        chat_model = None
        st.error(f"Model initialization error: {error}")
    
    # Chat history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_warnings" not in st.session_state:
        st.session_state["rag_warnings"] = []

    retriever = initialize_retriever() if enable_rag else None

    if enable_rag and st.session_state.get("rag_warnings"):
        for warning in st.session_state["rag_warnings"]:
            st.warning(warning)

    if enable_rag and retriever is None:
        st.info(
            f"No knowledge base loaded. Add .txt/.md/.pdf files in '{settings.knowledge_base_dir}' folder."
        )
    
    # Chat message to be displayed in the UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input of the chat
    if prompt := st.chat_input("Type your message here..."):
        # user messages are being added in the chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # user message are being displayed
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # chat bot responses are generated and displayed
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                rag_context = retrieve_context(retriever, prompt) if enable_rag else ""
                web_context = ""

                if enable_web_search and should_trigger_web_search(prompt):
                    web_results = search_web(prompt)
                    web_context = format_web_results(web_results)

                system_prompt = build_system_prompt(
                    response_mode=response_mode,
                    rag_context=rag_context,
                    web_context=web_context,
                )

                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)
        
        # chat history for the bot
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        if chat_model is None:
            st.info(" No valid provider/API key found. Please check the Instructions page and config setup.")

def main():
    st.set_page_config(
        page_title="NeoStats Insight Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    

    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
        if page == "Chat":
            st.divider()
            if st.button(" Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
