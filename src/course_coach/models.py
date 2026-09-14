"""One seam, many models — including one that costs nothing and runs offline.

Every client here answers the same two-argument call, so swapping a model is one
environment variable and never a code change:

    complete(system, user) -> str

WHY A SEAM AND NOT AN SDK. The point of this project is to compare harnesses,
and you cannot compare what you cannot swap. A provider is a base URL, a key and
a model name; all but one of these speak the OpenAI chat-completions shape, so
they are one client with different defaults rather than four integrations.

WHAT A MODEL CHANGE MAY MOVE, AND WHAT IT MAY NOT. It moves wording, latency,
cost and how often the answer is right. It does not move what the retriever
found, which passages reached the prompt, or whether a citation was verified --
those are yours, and they are where most of the headroom is. Measure before and
after with `course-coach measure` rather than trusting the size of the model.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class ModelError(Exception):
    """Raised when a provider is unreachable, unconfigured, or answers oddly."""


class Client(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class Provider:
    """Where a model lives and what it is called.

    `key_env` is the NAME of an environment variable, never a key. Nothing in
    this project reads a key from anywhere else, prints one, or puts one in an
    error message.
    """

    name: str
    base_url: str
    key_env: str
    default_model: str
    note: str = ""


#: The lanes, and every one of them is optional. `ollama` is the only one that
#: needs no account, no key and no network beyond localhost, which is why it is
#: the default and why the tests never touch any of the others.
PROVIDERS: dict[str, Provider] = {
    "ollama": Provider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        key_env="",
        default_model="qwen2.5:7b-instruct",
        note="local, free, offline; install from https://ollama.com then: ollama serve",
    ),
    "moonshot": Provider(
        name="moonshot",
        base_url="https://api.moonshot.ai/v1",
        key_env="MOONSHOT_API_KEY",
        # Deliberately empty: model ids move, and a wrong default fails in a way
        # that reads like a broken tool. Pass --model, or set COACH_MODEL.
        default_model="",
        note="OpenAI-compatible. Use api.moonshot.cn/v1 from mainland China.",
    ),
    "openai": Provider(
        name="openai",
        base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        default_model="",
    ),
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        default_model="",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        default_model="",
        note="many models behind one key; useful for comparing several at once",
    ),
}


class ChatClient:
    """Any OpenAI-shaped chat endpoint, over stdlib urllib. No SDK, no wheel.

    Temperature is 0 by default and that is not an accident: a coach that answers
    the same question two different ways cannot be measured, and a measurement
    you cannot repeat is an opinion.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout: float = 120,
        temperature: float = 0.0,
    ) -> None:
        if not model:
            raise ModelError("no model named: pass --model or set COACH_MODEL")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            }
        ).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self._key:
            headers["authorization"] = f"Bearer {self._key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            # The body can carry the provider's own explanation, which is worth
            # more than the status. It cannot carry our key: we never sent it
            # anywhere but the authorization header, and we do not echo that.
            detail = error.read().decode("utf-8", "replace").strip()[:200]
            raise ModelError(f"{self.base_url} refused with {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ModelError(f"no model server at {self.base_url}: {error}") from error
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ModelError("the provider answered without choices[0].message") from error


class EchoClient:
    """A client that answers from the passages alone, with no model at all.

    Not a mock: this is the default lane. Most questions a learner asks are
    answered by putting the right passage in front of them, and doing that needs
    no key, no download and no network. It also makes the retrieval quality
    visible instead of laundering it through fluent prose -- which is the first
    thing you have to see before you can improve it.
    """

    def complete(self, system: str, user: str) -> str:
        return ""


def get_client(provider: str = "", model: str = "", env: dict[str, str] | None = None) -> Client:
    """Build a client from a provider name, a model and the environment.

    Unconfigured means `EchoClient`, never an error: a learner with no key and no
    Ollama still gets a working coach, just one that quotes instead of talks.
    """
    environment = dict(os.environ) if env is None else dict(env)
    name = provider or environment.get("COACH_PROVIDER", "")
    if not name:
        return EchoClient()
    if name == "echo":
        return EchoClient()
    if name not in PROVIDERS:
        known = ", ".join(sorted([*PROVIDERS, "echo"]))
        raise ModelError(f"unknown provider {name!r}. Known: {known}")

    lane = PROVIDERS[name]
    key = environment.get(lane.key_env, "") if lane.key_env else ""
    if lane.key_env and not key:
        raise ModelError(f"{name} needs {lane.key_env} in the environment")
    chosen = model or environment.get("COACH_MODEL", "") or lane.default_model
    base_url = environment.get("COACH_BASE_URL", "") or lane.base_url
    return ChatClient(model=chosen, base_url=base_url, api_key=key)
