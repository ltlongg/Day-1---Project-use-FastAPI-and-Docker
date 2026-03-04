from transformers import pipeline


class TextClassifier:
    """Sentiment analysis model using Hugging Face Transformers."""

    def __init__(self):
        self.model = None
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    def load_model(self):
        """Load the pre-trained sentiment analysis model."""
        print(f"Loading model: {self.model_name}...")
        self.model = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            top_k=None,  # Return all scores
        )
        print("Model loaded successfully!")

    def predict(self, text: str) -> list[dict]:
        """
        Classify the input text.

        Args:
            text: Input text to classify.

        Returns:
            List of dicts with 'label' and 'score' keys.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model(text)
        # results is a list of list of dicts: [[{'label': ..., 'score': ...}, ...]]
        return results[0]


# Singleton instance
classifier = TextClassifier()
