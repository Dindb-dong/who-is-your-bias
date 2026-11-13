# Exact M7 orchestration source

`m7_delivery.py` is the exact operational helper preserved in the
[completed-29 preparation archive](../completed29-preparation.json).
Its SHA-256 is `1d543d3983fb0e1d2536eec8e041780a95cd6f49c2d5e7efd68ae7edeeceebf7`.
This copy makes the prepared execution and archive-verification logic reviewable
in Git. The original helper and its accompanying `bindings.json` remain in the
verified artifact; this source copy alone is not a replacement binding bundle.

The script retains run-specific paths and uses the existing shared supervisor.
It runs one explicitly chosen condition, then packages normal or failed harness
outputs. It does not automatically run another condition or publish success.
The original source plus exact bindings must be restored together before use.
For a supervisor interruption, inspect its actual terminal owner and harness
cleanup evidence before invoking the separate `package-run` path. Direct-child
exit alone does not establish cleanup of every service descendant.

The [portable runtime guide](../README.md) and frozen protocol remain unchanged.
Preparation loaded no neural weights. Actual invocation, saved output and
functional acceptance are separate evidence recorded by subsequent result files.
