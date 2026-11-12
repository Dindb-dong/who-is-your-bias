"""Verify named voice references without inferring identity from audio."""

from __future__ import annotations

import hashlib
from pathlib import Path


class ReferenceBindingError(ValueError):
    pass


def verify_binding(*, manifest_path: Path | None, persona_id: str | None,
                   speaker: str | None, audio_path: Path | None, text: str | None,
                   embedding_only: bool = False, allow_candidate: bool = False) -> dict:
    """Return bounded public identity; publisher labels are the identity basis.

    An unbound/generic historical service may still synthesize. A named service
    must bind the exact input bytes and preserve candidate/reviewed status.
    """
    if manifest_path is None:
        if persona_id not in {None, "generic"}:
            raise ReferenceBindingError("A named voice requires a source reference manifest")
        return {"persona_id": persona_id, "binding": "generic" if persona_id else "unbound",
                "review_status": "unverified", "synthesis": "reference_conditioned",
                "weight_adaptation": False}
    try:
        from tools.media.common import artifact_path, sha256
        from tools.references.workflow import verify_reference
        manifest = verify_reference(manifest_path)
        if not persona_id or persona_id == "generic" or not speaker:
            raise ReferenceBindingError("Configure the named persona and exact publisher speaker label")
        if manifest["publisher_attribution"]["speaker_label"] != speaker:
            raise ReferenceBindingError("Reference publisher label differs from the configured speaker")
        # A sidecar may strengthen the binding, but cannot silently override it.
        if manifest.get("persona_id") not in {None, persona_id}:
            raise ReferenceBindingError("Reference manifest belongs to a different persona")
        expected = {"jensen-huang": "Jensen Huang", "elon-musk": "Elon Musk"}.get(persona_id)
        if expected and speaker != expected:
            raise ReferenceBindingError("Configured publisher label differs from the named persona")
        artifacts = manifest["artifacts"]
        if audio_path is None or sha256(audio_path) != artifacts["audio"]["sha256"]:
            raise ReferenceBindingError("Configured reference audio differs from the verified artifact")
        if not embedding_only:
            original = artifact_path(manifest_path, artifacts["reference_text"]["path"]).read_text(encoding="utf-8").strip()
            if text is None or text.strip() != original:
                raise ReferenceBindingError("Configured reference text differs from the verified artifact")
        status = manifest["review"]["status"]
        if status != "reviewed" and not (status == "candidate" and allow_candidate):
            raise ReferenceBindingError("Reference is not reviewed; candidate experiments require explicit server opt-in")
        return {"persona_id": persona_id, "speaker_label": speaker, "binding": "publisher_attributed",
                "review_status": status, "reference_sha256": artifacts["audio"]["sha256"],
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_id": manifest["source_id"], "synthesis": "reference_conditioned",
                "weight_adaptation": False, "speaker_similarity_evaluated": False}
    except ReferenceBindingError:
        raise
    except (KeyError, ValueError, OSError, TypeError) as exc:
        raise ReferenceBindingError("The voice reference provenance could not be verified") from exc
