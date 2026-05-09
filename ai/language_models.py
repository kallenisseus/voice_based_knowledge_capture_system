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
        context: dict | None = None,
        mode: str = "full",
    ) -> dict:
        if not texts:
            return {
                "summary_sections": [],
                "summary": existing_summary,
            }

        normalized_mode = str(mode or "full").strip().lower()
        if normalized_mode not in {"full", "incremental"}:
            normalized_mode = "full"

        if normalized_mode == "incremental":
            mode_rules = (
                "- Update the existing summary with ONLY new facts explicitly present in NEW snippets.\n"
                "- Keep stable parts of the existing summary when NEW snippets do not change them.\n"
                "- Do not add external knowledge, assumptions, or inferred steps.\n"
                "- If NEW snippets are mostly intro/outro/engagement language, say that no meaningful update exists.\n"
                "- Include 1 to 4 sections depending on evidence density.\n"
                "- Each section body should be 1 to 3 sentences.\n"
                "- Keep summary concise (50 to 160 words).\n"
            )
            user_instruction = (
                f"Taxonomy context: {self._format_taxonomy_context(context)}\n"
                f"Existing summary baseline:\n{existing_summary or 'None'}\n\n"
                "Use existing summary only as baseline text. NEW snippets are the only new evidence.\n\n"
                "NEW snippets:\n"
                + "\n---\n".join(texts)
            )
        else:
            mode_rules = (
                "- Use ONLY facts explicitly present in the snippets.\n"
                "- Do not infer missing steps, tools, root causes, safety advice, or outcomes.\n"
                "- If snippets are mostly intro/outro or engagement talk, say that plainly.\n"
                "- Include 1 to 5 sections depending on available evidence.\n"
                "- Each section body should be 1 to 4 sentences.\n"
                "- Keep summary concise (50 to 180 words based on evidence density).\n"
            )
            user_instruction = (
                f"Taxonomy context: {self._format_taxonomy_context(context)}\n"
                "Do not use external knowledge or prior summaries.\n"
                "Only summarize what is explicitly stated below.\n\n"
                "New snippets:\n"
                + "\n---\n".join(texts)
            )

        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0.2,
            max_tokens=900,
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
                        + mode_rules +
                        "- Use complete sentences and avoid bullet-list formatting.\n"
                        "- Never output markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": user_instruction,
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

    def name_category(self, texts: list[str], existing_names: list[str], context: dict | None = None) -> str:
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
                    "content": (
                        f"Taxonomy context: {self._format_taxonomy_context(context)}\n\n"
                        + "\n---\n".join(texts)
                    ),
                },
            ],
        )

        return response.choices[0].message.content.strip()

    def _format_taxonomy_context(self, context: dict | None) -> str:
        """Serialize upload taxonomy metadata for prompt context."""
        if not context:
            return "None"

        machine_type = str(context.get("machine_type") or "").strip() or "Unassigned"
        machine_name = str(context.get("machine_name") or "").strip() or "Unknown Machine"
        subcategory_paths = []
        for raw_path in (context.get("subcategory_paths") or []):
            if isinstance(raw_path, str):
                parts = [part.strip() for part in raw_path.split(">")]
            elif isinstance(raw_path, (list, tuple)):
                parts = [str(value).strip() for value in raw_path]
            else:
                continue
            path = [part for part in parts if part]
            if path:
                subcategory_paths.append(path)

        hierarchy_raw = context.get("hierarchy_path") or []
        hierarchy_path = [str(value).strip() for value in hierarchy_raw if str(value).strip()]

        tags_raw = context.get("extra_tags") or []
        tags = [str(value).strip() for value in tags_raw if str(value).strip()]

        parts = [
            f"type={machine_type}",
            f"machine={machine_name}",
        ]
        if subcategory_paths:
            parts.append("subcategory_paths=" + "; ".join(" > ".join(path) for path in subcategory_paths[:8]))
        if hierarchy_path:
            parts.append("hierarchy=" + " > ".join(hierarchy_path))
        if tags:
            parts.append("tags=" + ", ".join(tags))
        return " | ".join(parts)
