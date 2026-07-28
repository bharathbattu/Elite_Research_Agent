class ResearchError(Exception):
    """Base exception for expected research failures."""


class ConfigurationError(ResearchError):
    """Required configuration is missing or invalid."""


class RetrievalError(ResearchError):
    """Evidence could not be retrieved."""


class ModelError(ResearchError):
    """The language model request or response failed."""


class ResearchQualityError(ResearchError):
    """Generated research did not meet evidence or citation requirements."""
