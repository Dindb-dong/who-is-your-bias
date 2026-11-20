"""Use the complete reply as Qwen3-TTS's text context before audio generation."""
from services.voice.app import create_app
from services.voice.engine import SeededQwen, VoiceSettings, load_qwen_model


class FullTextQwen(SeededQwen):
    def generate_voice_clone(self, **kwargs):
        kwargs["non_streaming_mode"] = True
        return super().generate_voice_clone(**kwargs)


def load_full_text_model(settings: VoiceSettings) -> FullTextQwen:
    seeded = load_qwen_model(settings)
    return FullTextQwen(seeded._qwen, seeded.torch, seeded.seed)


app = create_app(model_loader=load_full_text_model)
