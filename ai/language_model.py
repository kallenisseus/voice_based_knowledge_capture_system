import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class openai_client:
    """
    Hybrid client:
    - local embeddings via SentenceTransformers
    - online summaries/category names via OpenAI
    """

    def __init__(self):
        load_dotenv()

        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI()

        # Local multilingual embedding model
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Online chat model for naming/summaries
        self.chat_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def get_embeddings(self, texts):
        """
        Convert a list of text strings into embedding vectors locally.
        Returns a numpy array of shape (num_texts, embedding_dim).
        """
        if not texts:
            return np.array([])

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def get_embeddings_batched(self, texts, batch_size=100):
        """
        Local batched embeddings. Mostly useful if you later process many texts.
        """
        if not texts:
            return np.array([])

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def get_summary(self, texts):
        """
        Ask an online chat model to rewrite multiple texts into one coherent summary.
        """
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
                        "Focus on making the summary easy to read and understandable, but do not add any information that is not explicitly stated in the original texts."
                        "IMPORTANT: Skip information not related to the main topic. The summary should be concise but comprehensive."
                        "Split the text up in parts where you feel there is a natural break, add a row separator."
                        "If you find out important tools, parts exc, you can list them above under Tools Needed:"

                    ),
                },
                {
                    "role": "user",
                    "content": "\n---\n".join(texts),
                },
            ],
        )

        return response.choices[0].message.content.strip()

    def get_category(self, texts, previous):
        """
        Ask an online chat model for a short category/topic name.
        """
        if not texts:
            return "Empty"

        previous_str = json.dumps(previous, ensure_ascii=False)

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create a 1 to 3 word name for the topic of these texts. "
                        "Fewer words is better. Make it specific. "
                        "Do not collide with any of these existing category names: "
                        f"{previous_str}. "
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