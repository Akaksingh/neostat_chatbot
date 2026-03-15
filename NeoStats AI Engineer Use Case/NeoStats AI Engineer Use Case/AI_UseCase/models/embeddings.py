from langchain_huggingface import HuggingFaceEmbeddings

from config.config import get_settings


def get_embedding_model():
	try:
		settings = get_settings()
		return HuggingFaceEmbeddings(
			model_name=settings.embedding_model_name,
			model_kwargs={"device": "cpu"},
			encode_kwargs={"normalize_embeddings": True},
		)
	except Exception as error:
		raise RuntimeError(f"Failed to initialize embedding model: {error}")
