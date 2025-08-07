"""Sentiment Analyzer Pro: Production-Ready Sentiment Analysis with Photonic Neural Networks."""

__version__ = "1.0.0"
__author__ = "Terragon Labs"

# Core API exports - import lazily to avoid dependency issues
def __getattr__(name):
    """Lazy import to avoid dependency issues during development."""
    if name == "SentimentAnalyzer":
        from .analyzer import SentimentAnalyzer
        return SentimentAnalyzer
    elif name == "PhotonicSentimentModel":
        from .models import PhotonicSentimentModel
        return PhotonicSentimentModel
    elif name == "RealTimeSentimentProcessor":
        from .processor import RealTimeSentimentProcessor
        return RealTimeSentimentProcessor
    elif name == "SentimentBatchProcessor":
        from .processor import SentimentBatchProcessor
        return SentimentBatchProcessor
    elif name == "MultiLanguageSentimentAnalyzer":
        from .analyzer import MultiLanguageSentimentAnalyzer
        return MultiLanguageSentimentAnalyzer
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "__version__",
    "SentimentAnalyzer",
    "PhotonicSentimentModel",
    "RealTimeSentimentProcessor",
    "SentimentBatchProcessor", 
    "MultiLanguageSentimentAnalyzer",
]