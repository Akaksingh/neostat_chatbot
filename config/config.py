import os
from dataclasses import dataclass


@dataclass
class Settings:
	# currently we are have tested on groq api key only 
	default_provider: str = "groq"
	groq_api_key: str = ""
	groq_model: str = "llama-3.1-8b-instant"
	openai_api_key: str = ""
	openai_model: str = "gpt-4o-mini"
	gemini_api_key: str = ""
	gemini_model: str = "gemini-1.5-flash"
	tavily_api_key: str = ""
	embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
	knowledge_base_dir: str = "knowledge_base"


def get_settings() -> Settings:
	try:
		return Settings(
			default_provider=os.getenv("DEFAULT_LLM_PROVIDER", "groq").strip().lower(),
			groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
			groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip(),
			openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
			openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
			gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
			gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip(),
			tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
			embedding_model_name=os.getenv(
				"EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
			).strip(),
			knowledge_base_dir=os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base").strip(),
		)
	except Exception:
		return Settings()
