"""Local-only benchmark runner for consented-face-redactor.

All benchmark data is synthetic/local — no real biometric media,
model weights, secrets, or external downloads.
"""

__all__ = ["run_benchmark", "RunnerResult"]

from consented_face_redactor.benchmark.runner import run_benchmark, RunnerResult
