def build_system_prompt(response_mode: str, rag_context: str = "", web_context: str = "") -> str:
    try:
        style = (
            "You must answer in 3-5 bullet points or short paragraph summary."
            if response_mode == "Concise"
            else "You must answer with structured detail, include rationale, assumptions, and practical next steps."
        )

        policy = (
            "You are NeoStats Insight Assistant. Provide practical, accurate, domain-aware answers for analytics, product, and technology questions. "
            "If context is insufficient, clearly say what is missing instead of hallucinating."
        )

        context_sections = []
        if rag_context:
            context_sections.append(f"Knowledge Base Context:\n{rag_context}")
        if web_context:
            context_sections.append(f"Live Web Context:\n{web_context}")

        if context_sections:
            policy += "\nUse the supplied contexts as first-class evidence and cite source labels like [1], [2] when relevant."

        context_blob = "\n\n".join(context_sections)

        return f"{policy}\n{style}\n\n{context_blob}".strip()
    except Exception:
        return "You are a helpful assistant."
