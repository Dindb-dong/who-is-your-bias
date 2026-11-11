Generated voice samples for the 2026-11-11 CPU evaluation

Open listener/index.html after checking out the repository. The page works locally and needs no inference server. Share only listener/ for blinded listening; private/answer-key.json contains the condition, case names, and expected generated text.

These four PCM16 WAV files are the actual generated outputs from the saved Qwen3-TTS measurement, with original SHA-256 hashes. They total 1,348,020 bytes (approximately 1.29 MiB). No publisher/reference recording, reference transcript, trained checkpoint, or voice embedding is included. The optional reference-similarity score is therefore unavailable in this bundle.

All review fields begin unrated. Creating the bundle did not perform human listening. Validate an exported review with:

python -m tools.voice_evaluation review tools/voice_evaluation/results/qwen06-cpu-2026-11-11-listening/listener/manifest.json /path/to/listening-review.json

The main result is ../qwen06-cpu-2026-11-11.json. WER/CER measure disagreement with an independent recognizer; they do not establish voice likeness or naturalness. The first synthesis request includes lazy model loading. This is one seed, reference, and configuration with four samples.
