import json
import os
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ai import paths


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"


class KnowledgeModelClient:
    """
    Handles the model-facing parts of the workflow:
    - local embeddings for similarity search and clustering
    - OpenAI calls for category names and summaries
    """

    def __init__(self):
        # Keep runtime behavior aligned with Django settings: use the current
        # project `.env` file even if the shell already has older values set.
        load_dotenv(paths.PROJECT_ROOT / ".env", override=True)
        self.chat_model = os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.client = OpenAI()
        self._embedding_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        return self._embedding_model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def summarize_cluster(self, texts: list[str]) -> str:
        if not texts:
            return ""

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the contents of these texts into one summary of the content. "
                        "No information not contained in the original is allowed in the summary. "
                        "Focus on making the summary easy to read and understandable, but do not add any information that is not explicitly stated in the original texts. "
                        "Skip information not related to the main topic. The summary should be concise but comprehensive. "
                        "Split the text up in parts where you feel there is a natural break, add a row separator."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n---\n".join(texts),
                },
            ],
        )

        return response.choices[0].message.content.strip()

    def summarize_cluster_structured(
        self,
        texts: list[str],
        existing_summary: str = "",
    ) -> dict:
        if not texts:
            return {
                "summary_sections": [],
                "summary": existing_summary,
            }

        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0.2,
            max_tokens=1100,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are summarizing maintenance snippets.\n"
                        "Return ONLY valid JSON with this exact schema:\n"
                        "{"
                        "\"summary_sections\": ["
                        "{\"title\": \"...\", \"body\": \"...\"}"
                        "], "
                        "\"summary\": \"a detailed long-form summary\""
                        "}.\n"
                        "Rules:\n"
                        "- Keep language concrete and faithful to source text.\n"
                        "- Include 5 to 7 sections.\n"
                        "- Each section body should be 4 to 7 sentences.\n"
                        "- The summary field should be 180 to 260 words.\n"
                        "- Explain sequence, intent, and common failure points.\n"
                        "- Use complete sentences and avoid bullet-list formatting.\n"
                        "- Never output markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Existing summary: {existing_summary or 'None'}\n"
                        "Preserve useful details from the existing summary when still relevant.\n\n"
                        "New snippets:\n"
                        + "\n---\n".join(texts)
                    ),
                },
            ],
        )

        raw = response.choices[0].message.content.strip()
        payload = self._extract_json(raw)
        if not isinstance(payload, dict):
            raise ValueError("Structured summary response was not JSON.")

        sections = payload.get("summary_sections") or []
        normalized_sections = []
        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue
                title = str(section.get("title", "")).strip()
                body = str(section.get("body", "")).strip()
                if title and body:
                    normalized_sections.append({"title": title, "body": body})

        summary = str(payload.get("summary", "")).strip()
        if not summary and normalized_sections:
            summary = " ".join(section["body"] for section in normalized_sections[:2]).strip()

        return {
            "summary_sections": normalized_sections,
            "summary": summary,
        }

    def _extract_json(self, text: str):
        candidate = text.strip()

        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model response.")
        return json.loads(match.group(0))

    def name_category(self, texts: list[str], existing_names: list[str]) -> str:
        if not texts:
            return "Empty"

        existing_names_json = json.dumps(existing_names, ensure_ascii=False)

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create a 1 to 3 word name for the topic of these texts. "
                        "Fewer words is better. Make it specific. "
                        "Do not collide with any of these existing category names: "
                        f"{existing_names_json}. "
                        "Return only the category name."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n---\n".join(texts),
                },
            ],
        )

        return response.choices[0].message.content.strip()
