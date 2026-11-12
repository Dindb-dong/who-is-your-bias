"""Keep source, speech and motion identities attached to their active persona.

These are operator/curator declarations, not biometric identity inference.
An old reference-voice endpoint must not silently become the new subject's voice.
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import ProviderSettings, Settings
from .corpus import Corpus, source_event_id
from .motion import MotionLibrary
from .providers import ProviderError, SpeechProvider


def _media_url(value: str) -> str:
    parsed = urlsplit(value)
    # The frozen TED corpus links the transcript view of the same recording.
    # Do not drop content-bearing query parameters such as a YouTube video ID.
    query = [(key, item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)
             if not (parsed.hostname in {"ted.com", "www.ted.com"} and key == "view" and item == "transcript")]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), urlencode(sorted(query)), ""))


def motion_binding_status(library: MotionLibrary, corpus: Corpus) -> str:
    """Accept a declared subject, or unchanged legacy recordings in this corpus.

    New libraries can declare their subject separately from the factual corpus.
    Legacy catalogs have no subject field; every recording must already belong
    to the active corpus with matching event and media URL before playback.
    """
    if library.catalog.persona_id is not None:
        return "matched" if library.catalog.persona_id == corpus.persona.id else "subject_mismatch"
    if not library.catalog.sources:
        return "unbound"
    for item in library.catalog.sources:
        source = corpus.sources.get(item.id)
        if source is None or source_event_id(source) != item.source_event_id or _media_url(str(source.url)) != _media_url(item.url):
            return "subject_mismatch"
    return "source_bound"


def voice_binding(settings: Settings, persona_id: str) -> dict[str, str | None]:
    if not settings.tts.configured:
        status = "not_configured"
    elif settings.tts_persona_id is None:
        status = "unbound"
    elif settings.tts_persona_id == "generic":
        status = "generic"
    elif settings.tts_persona_id == persona_id:
        status = "matched"
    else:
        status = "subject_mismatch"
    return {"status": status, "persona_id": settings.tts_persona_id, "voice": settings.tts_voice}


class SubjectSpeechProvider(SpeechProvider):
    """One declared voice for a session; callers cannot override its identity."""

    def __init__(self, config: Settings, persona_id: str):
        self.binding = voice_binding(config, persona_id)
        self.provider_configured = config.tts.configured
        enabled = self.binding["status"] in {"matched", "generic"}
        # Live streaming checks settings.configured before starting generation.
        super().__init__(config.tts if enabled else ProviderSettings(), voice=config.tts_voice,
                         timeout=config.request_timeout, language_hint=config.tts_language_hint)

    async def synthesize(self, text: str, voice: str | None = None, *, language: str | None = None) -> tuple[bytes, str]:
        if self.provider_configured and self.binding["status"] not in {"matched", "generic"}:
            raise ProviderError("speech_subject_mismatch", "Bind the configured voice to this persona, or explicitly select a generic voice, before playback.", 503)
        if voice is not None and voice != self.voice:
            raise ProviderError("speech_voice_override", "This session uses its configured voice. Select another voice in server configuration with its matching persona binding.", 422)
        return await super().synthesize(text, voice, language=language)
