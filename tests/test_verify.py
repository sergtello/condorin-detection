from __future__ import annotations

import json

import requests
from test_job import BOUNDS, SCENE, VESSEL, RecordingReporter, StubDetector, put, settings_for

from condorin.job.config import JobSettings
from condorin.job.runner import Runner
from condorin.job.verify import JevVerifier, build_verifier


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self.body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class FakeSession:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append((url, json, headers, timeout))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def jev_answer(other: float) -> FakeResponse:
    return FakeResponse(200, {"answers": {"objeto": {
        "type": "choice", "choice": "otra_cosa" if other >= 0.5 else "draga", "confidence": 0.9,
        "probabilities": {"draga": 1 - other, "otra_cosa": other}}}})  # fmt: skip


def verifier(result, **kw) -> tuple[JevVerifier, FakeSession]:
    session = FakeSession(result)
    return JevVerifier(JobSettings(jev_api_key="k", **kw), session), session


def run_with(tmp_path, card, v):
    put(tmp_path, SCENE, {"bounds": json.dumps(BOUNDS)})
    reporter = RecordingReporter()
    r = Runner(settings_for(tmp_path), card, StubDetector(card, [VESSEL]), reporter, v)
    return r.run([SCENE]), reporter.sent[0][0]


def test_off_without_a_key():
    assert build_verifier(JobSettings()) is None
    assert isinstance(build_verifier(JobSettings(jev_api_key="k")), JevVerifier)


def test_request_carries_the_crop_and_a_choice_question():
    v, session = verifier(jev_answer(0.1))
    assert v.rejects(b"\x89PNG") is False
    url, body, headers, timeout = session.calls[0]
    assert url == "https://openrouter.ai/api/alpha/decisions" and timeout == 10.0
    assert (
        headers["Authorization"] == "Bearer k" and headers["TypeSafe-Danger-Zone-Images"] == "true"
    )
    assert body["model"] == "typesafe/jev-1.14"
    assert body["state"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    q = body["questions"]["objeto"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"draga", "otra_cosa"}


def test_confident_false_positive_is_dropped_and_the_scene_reported_empty(tmp_path, card):
    v, _ = verifier(jev_answer(0.95))
    summary, payload = run_with(tmp_path, card, v)
    assert payload["detections"] == []
    assert (summary.processed, summary.with_findings, summary.without_findings) == (1, 0, 1)


def test_unsure_jev_keeps_the_box(tmp_path, card):
    v, _ = verifier(jev_answer(0.6))  # below JEV_REJECT_MIN 0.8
    _, payload = run_with(tmp_path, card, v)
    assert len(payload["detections"]) == 1


def test_any_jev_failure_keeps_the_box(tmp_path, card):
    for result in (requests.Timeout("slow"), FakeResponse(500, {}), FakeResponse(200, {"answers": {}}),
                   FakeResponse(200, ValueError("not json"))):  # fmt: skip
        v, _ = verifier(result)
        assert v.rejects(b"png") is False
    v, _ = verifier(requests.ConnectionError("down"))
    _, payload = run_with(tmp_path, card, v)
    assert len(payload["detections"]) == 1
