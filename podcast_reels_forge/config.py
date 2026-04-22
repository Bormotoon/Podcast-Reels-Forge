"""Configuration helpers for Podcast Reels Forge.

This module keeps the role-based Ollama model mapping and the filesystem-safe
model folder naming rules in one place so the pipeline, analyzer and docs stay
consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ALLOWED_OLLAMA_MODELS = frozenset(
    {
        "gemma4:26b",
        "gemma4:e4b",
        "gemma3:12b",
        "gemma3:4b",
    },
)

DEFAULT_ROLE_ORDER = ("scout", "cleanup", "refine", "judge")


@dataclass(frozen=True)
class OllamaRoleMapping:
    """Resolved model roles used by the default analysis pipeline."""

    scout: str
    cleanup: str
    refine: str
    judge: str
    metadata: str

    def as_dict(self) -> dict[str, str]:
        return {
            "scout": self.scout,
            "cleanup": self.cleanup,
            "refine": self.refine,
            "judge": self.judge,
            "metadata": self.metadata,
        }

    def unique_models(self) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for model in self.as_dict().values():
            if model not in seen:
                seen.add(model)
                out.append(model)
        return tuple(out)


def normalize_model_folder_name(model: str) -> str:
    """Return the deterministic filesystem-safe folder name for a model id."""

    normalized = (model or "").strip().lower()
    special = {
        "gemma4:26b": "gemma4_26b",
        "gemma4:e4b": "gemma4_e4b",
        "gemma3:12b": "gemma3_12b",
        "gemma3:4b": "gemma3_4b",
    }
    if normalized in special:
        return special[normalized]

    out: list[str] = []
    for ch in normalized:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        elif ch in {":", ".", "/", " "}:
            out.append("_")
    folder = "".join(out).strip("_")
    return folder or "model"


def _clean_model_name(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _load_role_model_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        role = _clean_model_name(key).lower()
        model = _clean_model_name(value)
        if role and model:
            out[role] = model
    return out


def _validate_allowed_models(models: Mapping[str, str]) -> None:
    invalid = sorted(
        {
            model
            for model in models.values()
            if model and model not in ALLOWED_OLLAMA_MODELS
        },
    )
    if invalid:
        allowed = ", ".join(sorted(ALLOWED_OLLAMA_MODELS))
        raise SystemExit(
            "Unsupported Ollama models in config. Allowed default lineup: "
            f"{allowed}; got: {', '.join(invalid)}",
        )


def resolve_ollama_role_mapping(conf: Mapping[str, Any] | None) -> OllamaRoleMapping:
    """Resolve the active role mapping from config.

    The preferred format is:

    ollama:
      roles:
        scout: gemma4:e4b
        cleanup: gemma3:4b
        refine: gemma3:12b
        judge: gemma4:26b

    A legacy `models:` list is still accepted for compatibility when the role
    map is absent, but the new role-based format is the default.
    """

    ollama_conf: Mapping[str, Any] = conf.get("ollama", {}) if isinstance(conf, Mapping) else {}
    if not isinstance(ollama_conf, Mapping):
        ollama_conf = {}

    role_map = _load_role_model_map(ollama_conf.get("roles"))
    if not role_map:
        legacy_models = [
            _clean_model_name(model)
            for model in ollama_conf.get("models", [])
            if _clean_model_name(model)
        ]
        if len(legacy_models) < len(DEFAULT_ROLE_ORDER):
            raise SystemExit(
                "Missing ollama.roles config. Provide the new role mapping or a "
                f"legacy models list with at least {len(DEFAULT_ROLE_ORDER)} models.",
            )
        role_map = {
            role: legacy_models[idx]
            for idx, role in enumerate(DEFAULT_ROLE_ORDER)
        }

    missing = [role for role in DEFAULT_ROLE_ORDER if role not in role_map]
    if missing:
        raise SystemExit(
            "Missing required ollama role assignments: " + ", ".join(missing),
        )

    scout = role_map["scout"]
    cleanup = role_map["cleanup"]
    refine = role_map["refine"]
    judge = role_map["judge"]
    metadata = role_map.get("metadata", judge)

    resolved = OllamaRoleMapping(
        scout=scout,
        cleanup=cleanup,
        refine=refine,
        judge=judge,
        metadata=metadata,
    )

    required_unique = [resolved.scout, resolved.cleanup, resolved.refine, resolved.judge]
    if len(set(required_unique)) != len(required_unique):
        raise SystemExit(
            "ollama.roles must assign four distinct models to scout, cleanup, refine, and judge",
        )

    _validate_allowed_models(
        {
            "scout": resolved.scout,
            "cleanup": resolved.cleanup,
            "refine": resolved.refine,
            "judge": resolved.judge,
            "metadata": resolved.metadata,
        },
    )

    return resolved


def merge_ollama_role_conf(
    base_conf: Mapping[str, Any] | None,
    model: str,
    *,
    role: str | None = None,
) -> dict[str, Any]:
    """Merge global Ollama config with role/model specific overrides."""

    base = dict(base_conf or {})
    merged: dict[str, Any] = dict(base)

    overrides = base.get("role_overrides")
    if role and isinstance(overrides, Mapping):
        role_override = overrides.get(role)
        if isinstance(role_override, Mapping):
            merged.update(role_override)

    model_overrides = base.get("model_overrides")
    if isinstance(model_overrides, Mapping):
        model_override = model_overrides.get(model)
        if isinstance(model_override, Mapping):
            merged.update(model_override)

    return merged
