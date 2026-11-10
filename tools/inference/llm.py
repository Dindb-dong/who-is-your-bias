"""Run one bounded CPU generation using the app's actual retrieval and prompt."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from bias.corpus import Corpus
from bias.engine import validate_citations
from bias.models import ChatRequest
from bias.prompts import build_messages
from bias.retrieval import Retriever

from .common import new_report, progress, safe_error, save_report


def run(args: argparse.Namespace) -> dict:
    report = new_report("local_llm")
    report.update(model_id=args.model, device="cpu", dtype=args.dtype, question=args.question,
                  language=args.language, mode=args.mode, max_new_tokens=args.max_new_tokens,
                  thinking=False, seed=args.seed, cpu_threads=args.threads)
    save_report(args.report, report)
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextStreamer

        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        corpus = Corpus.load(args.corpus)
        request = ChatRequest(message=args.question, language=args.language, mode=args.mode)
        retrieval_start = time.perf_counter()
        result = Retriever(corpus).search(request.message, limit=4, behavior=request.mode == "behavior")
        report["retrieval_seconds"] = time.perf_counter() - retrieval_start
        report["retrieved_ids"] = [hit.evidence.id for hit in result.hits]
        report["retrieval_confidence"] = result.confidence
        if not result.hits:
            raise RuntimeError("The selected question retrieved no reviewed evidence; choose a grounded smoke question")
        messages = build_messages(corpus.persona, request, result, [])
        report["corpus_sha256"] = hashlib.sha256((args.corpus / "sources.json").read_bytes() + (args.corpus / "evidence.jsonl").read_bytes()).hexdigest()
        progress("retrieved", evidence_ids=report["retrieved_ids"])

        info = HfApi(token=False).model_info(args.model, revision=args.revision, files_metadata=True, timeout=30)
        report["model_revision"] = info.sha
        weight_bytes = sum(file.size or 0 for file in info.siblings if file.rfilename.endswith(".safetensors"))
        report["published_safetensors_bytes"] = weight_bytes
        if not weight_bytes or weight_bytes > 2_500_000_000:
            raise RuntimeError("This smoke runner accepts only models with known safetensors weights up to 2.5 GB")
        save_report(args.report, report)
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=info.sha, cache_dir=args.cache_dir, token=False, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=info.sha, cache_dir=args.cache_dir, token=False,
            trust_remote_code=False, use_safetensors=True, device_map="cpu",
            dtype=getattr(torch, args.dtype), attn_implementation="sdpa", low_cpu_mem_usage=True,
        )
        model.eval()
        report["load_seconds"] = time.perf_counter() - load_start
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt")
        report["input_tokens"] = int(inputs.input_ids.shape[-1])
        report["actual_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        progress("model_loaded", seconds=report["load_seconds"], input_tokens=report["input_tokens"])
        save_report(args.report, report)

        started = time.perf_counter()
        first_chunk: float | None = None

        class Deadline(StoppingCriteria):
            expired = False

            def __call__(self, input_ids, scores, **kwargs):
                self.expired = time.perf_counter() - started >= args.timeout_seconds
                return self.expired

        class MeasuredStreamer(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                nonlocal first_chunk
                if text:
                    if first_chunk is None:
                        first_chunk = time.perf_counter() - started
                    progress("text_chunk", text=text)

        deadline = Deadline()
        streamer = MeasuredStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.8, top_k=20, streamer=streamer,
                stopping_criteria=StoppingCriteriaList([deadline]),
            )
        generation_seconds = time.perf_counter() - started
        tokens = output[0, inputs.input_ids.shape[-1]:]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        cleaned, valid, invalid = validate_citations(text, set(report["retrieved_ids"]))
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        report.update(
            status="timed_out" if deadline.expired else "completed",
            generation_seconds=generation_seconds, first_text_chunk_seconds=first_chunk,
            generated_tokens=int(tokens.numel()), tokens_per_second=float(tokens.numel() / generation_seconds),
            ended_with_eos=bool(tokens.numel() and int(tokens[-1]) in eos_ids),
            generated_text=text, display_text=cleaned,
            citation_check={"valid_ids": valid, "invalid_ids": invalid, "has_valid_citation": bool(valid), "claim_entailment": "not_automatically_validated"},
            persona_fidelity="not_evaluated",
        )
        progress("completed", seconds=generation_seconds, tokens=int(tokens.numel()), valid_citations=valid, invalid_citations=invalid)
    except Exception as error:
        report.update(status="failed", error=safe_error(error))
        progress("failed", error=report["error"])
    save_report(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", help="Optional immutable Hugging Face commit from a previous report")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--corpus", type=Path, default=Path("data/personas/elon-musk"))
    parser.add_argument("--question", default="Why did Tesla start with an expensive sports car?")
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--mode", choices=("rag", "behavior"), default="behavior")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/local/llm-smoke.json"))
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 160 or not 1 <= args.timeout_seconds <= 300 or not 1 <= args.threads <= 8:
        parser.error("Use 1–160 output tokens, 1–300 seconds, and 1–8 CPU threads")
    return 0 if run(args)["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
