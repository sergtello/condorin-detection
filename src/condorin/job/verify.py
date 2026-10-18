"""Second opinion on each crop: TypeSafe Jev (a decision model) says whether it is a dredge or something else.

Off unless `JEV_API_KEY` is set. It only drops a box when Jev is confident it is something else
(`JEV_REJECT_MIN`). Any error, timeout or unexpected answer keeps the box: a broken verifier must never
hide a real dredge, so the detector's result stands.
"""

from __future__ import annotations

import base64
import logging

import requests

from condorin.job.config import JobSettings

log = logging.getLogger(__name__)

QUESTION = "objeto"
DREDGE = "draga"
OTHER = "otra_cosa"
CRITERIA = {
    DREDGE: "Una draga o balsa de minería aurífera sobre un río: plataforma flotante con maquinaria, "
    "a veces con una pluma de sedimento.",
    OTHER: "Cualquier otra cosa: roca, tronco, banco de arena, bote común, casa, nube, sombra o ruido.",
}


class JevVerifier:
    def __init__(self, settings: JobSettings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()

    def other_probability(self, png: bytes) -> float | None:
        """Jev's probability that the crop is not a dredge, or None when Jev gave no usable answer."""
        s = self.settings
        body = {
            "model": s.jev_model,
            "state": [
                {"type": "text", "text": "Recorte de una imagen satelital de un río amazónico."},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            ],
            "questions": {QUESTION: {
                "type": "choice",
                "instructions": "¿Qué hay en el centro de este recorte?",
                "criteria": CRITERIA,
            }},
        }  # fmt: skip
        headers = {
            "Authorization": f"Bearer {s.jev_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
        }
        try:
            resp = self.session.post(s.jev_url, json=body, headers=headers, timeout=s.jev_timeout_s)
            resp.raise_for_status()
            answer = resp.json()["answers"][QUESTION]
            return float(answer["probabilities"][OTHER])
        except (requests.RequestException, ValueError, KeyError, TypeError) as err:
            log.warning(
                "jev unavailable, keeping the box", extra={"error": f"{type(err).__name__}: {err}"}
            )
            return None

    def rejects(self, png: bytes) -> bool:
        p = self.other_probability(png)
        return p is not None and p >= self.settings.jev_reject_min


def build_verifier(settings: JobSettings) -> JevVerifier | None:
    return JevVerifier(settings) if settings.jev_api_key else None
