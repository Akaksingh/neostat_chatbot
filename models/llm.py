from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional

from config.config import get_settings


def get_chat_model(provider: Optional[str] = None):
    """Initialize and return a chat model for the selected provider."""
    try:
        settings = get_settings()
        selected_provider = (provider or settings.default_provider).strip().lower()

        if selected_provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            return ChatOpenAI(api_key=settings.openai_api_key, model=settings.openai_model, temperature=0.2)

        if selected_provider == "gemini":
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set.")
            return ChatGoogleGenerativeAI(
                google_api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                temperature=0.2,
            )

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model, temperature=0.2)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize chat model: {str(e)}")


def get_available_providers() -> list[str]:
    try:
        settings = get_settings()
        providers = []
        if settings.groq_api_key:
            providers.append("groq")
        if settings.openai_api_key:
            providers.append("openai")
        if settings.gemini_api_key:
            providers.append("gemini")
        return providers
    except Exception:
        return []