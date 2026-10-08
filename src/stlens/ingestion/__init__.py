"""Raw-message boundary, raw/quarantine persistence and exchange adapters (Phase 1).

Exchange-specific code lives in one subpackage per exchange (``binance``); everything it
emits is a canonical event from :mod:`stlens.schemas`.
"""

from stlens.ingestion.raw import QuarantinedMessage, RawMessage, RawMessageError

__all__ = ["QuarantinedMessage", "RawMessage", "RawMessageError"]
