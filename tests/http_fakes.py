"""Stand-ins for ``requests`` sessions, so data downloaders are tested without a network."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class FakeResponse:
    status_code: int = 200
    content: bytes = b""
    headers: dict = field(default_factory=dict)
    payload: dict | None = None

    @property
    def text(self) -> str:
        return json.dumps(self.payload) if self.payload is not None else self.content.decode()

    def json(self) -> dict:
        return self.payload or {}


class FakeSession:
    """Answers each ``get`` with ``handler(url, params)``: a response or an exception to raise."""

    def __init__(self, handler: Callable[[str, dict | None], FakeResponse | Exception]) -> None:
        self.handler = handler
        self.calls: list[tuple[str, dict | None, dict | None]] = []

    def get(self, url: str, params: dict | None = None, headers: dict | None = None,
            timeout: object = None) -> FakeResponse:  # fmt: skip
        self.calls.append((url, dict(params) if params else None, headers))
        answer = self.handler(url, params)
        if isinstance(answer, Exception):
            raise answer
        return answer
