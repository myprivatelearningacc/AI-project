import os
from typing import Optional


class Generator:
    """
    Generate an answer from a prompt.

    If OPENAI_API_KEY is available, use OpenAI.
    Otherwise, return a simple fallback answer based on retrieved context.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, prompt: str, fallback_context: Optional[str] = None) -> str:
        """
        Generate answer using OpenAI if possible.
        Otherwise use fallback.
        """

        if self.api_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.api_key)

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful quantitative finance tutor."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2
                )

                return response.choices[0].message.content

            except Exception as e:
                return (
                    "OpenAI generation failed. Falling back to context-only answer.\n\n"
                    f"Error: {e}\n\n"
                    f"{fallback_context or ''}"
                )

        return (
            "No OpenAI API key found. Here is the most relevant retrieved context:\n\n"
            f"{fallback_context or ''}"
        )