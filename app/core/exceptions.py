class DocumentProcessingError(Exception):
    """Falha ao carregar, validar ou processar um documento."""


class ServiceUnavailableError(Exception):
    """Serviço essencial indisponível."""
