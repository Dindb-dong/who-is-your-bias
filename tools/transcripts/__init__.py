"""Publisher transcript acquisition; no speech models load at import time."""

from .ted import TranscriptError, extract_ted, read_document, write_bundle

__all__ = ["TranscriptError", "extract_ted", "read_document", "write_bundle"]
