# Explicit Jensen runtime candidate

This opt-in configuration binds the restored original-dialogue-v4 corpus, the
three-event motion catalog and the source-fitted avatar. The application default
is unchanged. It delivers original interview excerpts in **evidence-preview**
mode; it does not run a language model or generate a new persona response.

The separate runtime snapshots retain the frozen M6 corpus digest and cutoff.
They label the actual preview, optional voice and avatar roles without changing
the condition-neutral M6 comparison snapshot. Historical interviews do not
establish present-day positions. The optional retrieval candidate did not meet
the frozen retrieval acceptance gate, the motion remains `geometry_preview`,
the avatar has not passed human likeness review, and the 16-step voice adapter
requires separate listening evaluation. None is silently promoted here.

## Launch

Restore the durable artifacts described by the replication and modality reports,
install the locked application dependencies and build the web app with
`npm ci` and `npm run build`. From the integrated repository:

```bash
PYTHONPATH=src:. .venv/bin/python -m tools.runtime_checks verify \
  --artifacts /absolute/path/to/jensen-2026-11-13 --allow-candidates

PYTHONPATH=src:. .venv/bin/python -m tools.runtime_checks serve \
  --artifacts /absolute/path/to/jensen-2026-11-13 --allow-candidates
```

Open `http://127.0.0.1:8000`, then select **Jensen source-fitted candidate** in
Studio settings. The launcher checks the exact corpus, snapshot, motion catalog,
source assets and built avatar bytes before starting. It does not inherit a
language model, speech provider or persona snapshot from unrelated environment
settings.

Add `--with-voice --inference-python /absolute/path/to/.venv-inference/bin/python`
to start the separately pinned candidate voice service as a child process. This
loads the actual model on the first speech request and needs sufficient memory;
HTTP health before synthesis is not a completed model-load or quality check.
The voice runtime binding and reference verification are owned by
`tools.voice_evaluation.serve_candidate`. The launcher keeps API, provider and
test clients in the same process tree because this execution environment does
not share loopback networking across independent command sessions.

## Frozen functional protocol

`protocol.json` was written before execution. Its real TCP checks cover source
inspection, unchanged citation content, four same-session turns, current and
private-information controls, endorsement, Korean preview, SSE and disconnect
cleanup. Timings in this configuration measure evidence delivery and API work,
not language-model inference. The optional short speech probe is an integration
check excluded from the frozen voice quality comparison.

```bash
PYTHONPATH=src:. .venv/bin/python -m tools.runtime_checks smoke \
  --artifacts /absolute/path/to/jensen-2026-11-13 --allow-candidates \
  --output /absolute/path/outside-the-repository/runtime-preview-check
```

The runner starts the real API on an ephemeral loopback TCP port, executes the
protocol, writes raw HTTP evidence and a source-text-free report, and stops its
servers. The output directory must be new and outside the repository because
the raw trace contains original interview answers. Its report does not claim
that a quickly completed preview was interrupted neural inference.

Status: configuration preflight and Python compilation passed; the frozen
real HTTP runner is ready. Functional execution results are pending.
