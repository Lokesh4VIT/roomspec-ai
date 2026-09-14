"""Minimal Prometheus metrics without extra dependencies (single-process registry)."""

from __future__ import annotations

import threading
from collections import defaultdict

LATENCY_BUCKETS_S = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class _Histogram:
    def __init__(self):
        self.buckets = [0] * len(LATENCY_BUCKETS_S)
        self.count = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.sum += value
        for i, bound in enumerate(LATENCY_BUCKETS_S):
            if value <= bound:
                self.buckets[i] += 1


class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self.request_latency: dict[tuple[str, str], _Histogram] = defaultdict(_Histogram)
        self.stage_latency: dict[str, _Histogram] = defaultdict(_Histogram)
        self.spec_status: dict[str, int] = defaultdict(int)
        self.llm: dict[tuple[str, str], int] = defaultdict(int)

    def observe_request(self, method: str, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(method, route, str(status))] += 1
            self.request_latency[(method, route)].observe(seconds)

    def observe_spec(self, spec: dict) -> None:
        with self._lock:
            self.spec_status[spec["status"]] += 1
            for stage, ms in spec["timings_ms"].items():
                if stage != "total":
                    self.stage_latency[stage].observe(ms / 1000)
            h = spec["hallucination_check"]
            self.llm[(h["provider"], "used" if h["llm_used"] else "fallback")] += 1

    def render(self) -> str:
        def labels(**kv: str) -> str:
            return "{" + ",".join(f'{k}="{v}"' for k, v in kv.items()) + "}"

        def hist(name: str, h: _Histogram, **kv: str) -> list[str]:
            out = [
                f"{name}_bucket{labels(**kv, le=str(b))} {c}" for b, c in zip(LATENCY_BUCKETS_S, h.buckets, strict=True)
            ]
            out += [
                f"{name}_bucket{labels(**kv, le='+Inf')} {h.count}",
                f"{name}_sum{labels(**kv)} {h.sum:.6f}",
                f"{name}_count{labels(**kv)} {h.count}",
            ]
            return out

        with self._lock:
            lines = [
                "# HELP roomspec_http_requests_total HTTP requests by route and status.",
                "# TYPE roomspec_http_requests_total counter",
            ]
            lines += [
                f"roomspec_http_requests_total{labels(method=m, route=r, status=s)} {n}"
                for (m, r, s), n in sorted(self.requests.items())
            ]
            lines += [
                "# HELP roomspec_http_request_duration_seconds HTTP request latency.",
                "# TYPE roomspec_http_request_duration_seconds histogram",
            ]
            for (m, r), h in sorted(self.request_latency.items()):
                lines += hist("roomspec_http_request_duration_seconds", h, method=m, route=r)
            lines += [
                "# HELP roomspec_pipeline_stage_seconds Spec pipeline latency per stage.",
                "# TYPE roomspec_pipeline_stage_seconds histogram",
            ]
            for stage, h in sorted(self.stage_latency.items()):
                lines += hist("roomspec_pipeline_stage_seconds", h, stage=stage)
            lines += [
                "# HELP roomspec_specs_total Generated specifications by status.",
                "# TYPE roomspec_specs_total counter",
            ]
            lines += [f"roomspec_specs_total{labels(status=s)} {n}" for s, n in sorted(self.spec_status.items())]
            lines += [
                "# HELP roomspec_llm_narratives_total Narratives by provider and outcome.",
                "# TYPE roomspec_llm_narratives_total counter",
            ]
            lines += [
                f"roomspec_llm_narratives_total{labels(provider=p, outcome=o)} {n}"
                for (p, o), n in sorted(self.llm.items())
            ]
            return "\n".join(lines) + "\n"


registry = Registry()
