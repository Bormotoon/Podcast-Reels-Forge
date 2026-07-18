"""Configuration helpers for Podcast Reels Forge.

This module keeps the role-based llama.cpp model mapping and the filesystem-safe
model folder naming rules in one place so the pipeline, analyzer and docs stay
consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ALLOWED_LLAMA_CPP_MODELS = frozenset(
    {
        "gemma4",
        "gemma4:26b",
        "gemma4:27b",
        "gemma4:12b",
    },
)

DEFAULT_ROLE_ORDER = ("scout", "cleanup_refine", "judge_metadata")


@dataclass(frozen=True)
class LlamaCppRoleMapping:
    """Resolved model roles used by the default analysis pipeline."""

    scout: str
    cleanup_refine: str
    judge_metadata: str
    # Optional transcript proofreading role; falls back to cleanup_refine.
    proofread: str = "gemma4:26b"

    def as_dict(self) -> dict[str, str]:
        return {
            "scout": self.scout,
            "cleanup_refine": self.cleanup_refine,
            "judge_metadata": self.judge_metadata,
            "proofread": self.proofread,
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
        "gemma4": "gemma4",
        "gemma4:26b": "gemma4_26b",
        "gemma4:27b": "gemma4_27b",
        "gemma4:12b": "gemma4_12b",
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
            if model and model not in ALLOWED_LLAMA_CPP_MODELS
        },
    )
    if invalid:
        allowed = ", ".join(sorted(ALLOWED_LLAMA_CPP_MODELS))
        raise SystemExit(
            "Unsupported llama.cpp models in config. Allowed default lineup: "
            f"{allowed}; got: {', '.join(invalid)}",
        )


def resolve_llama_cpp_role_mapping(conf: Mapping[str, Any] | None) -> LlamaCppRoleMapping:
    """Resolve the active role mapping from config.

    The preferred format is:

        llama_cpp:
      roles:
                scout: gemma4
                cleanup: gemma4
                refine: gemma4
                judge: gemma4

    A legacy `models:` list is still accepted for compatibility when the role
    map is absent, but the new role-based format is the default.
    """

    llama_cpp_conf: Mapping[str, Any] = conf.get("llama_cpp", {}) if isinstance(conf, Mapping) else {}
    if not isinstance(llama_cpp_conf, Mapping):
        llama_cpp_conf = {}

    role_map = _load_role_model_map(llama_cpp_conf.get("roles"))
    if not role_map:
        legacy_models = [
            _clean_model_name(model)
            for model in llama_cpp_conf.get("models", [])
            if _clean_model_name(model)
        ]
        if len(legacy_models) < len(DEFAULT_ROLE_ORDER):
            raise SystemExit(
                "Missing llama_cpp.roles config. Provide the new role mapping or a "
                f"legacy models list with at least {len(DEFAULT_ROLE_ORDER)} models.",
            )
        role_map = {
            role: legacy_models[idx]
            for idx, role in enumerate(DEFAULT_ROLE_ORDER)
        }

    missing = [role for role in DEFAULT_ROLE_ORDER if role not in role_map]
    if missing:
        raise SystemExit(
            "Missing required llama_cpp role assignments: " + ", ".join(missing),
        )

    scout = role_map["scout"]
    cleanup_refine = role_map["cleanup_refine"]
    judge_metadata = role_map["judge_metadata"]
    # Proofread is optional: older configs without it reuse the cleanup model.
    proofread = role_map.get("proofread") or cleanup_refine

    resolved = LlamaCppRoleMapping(
        scout=scout,
        cleanup_refine=cleanup_refine,
        judge_metadata=judge_metadata,
        proofread=proofread,
    )

    _validate_allowed_models(
        {
            "scout": resolved.scout,
            "cleanup_refine": resolved.cleanup_refine,
            "judge_metadata": resolved.judge_metadata,
            "proofread": resolved.proofread,
        },
    )

    return resolved


def merge_llama_cpp_role_conf(
    base_conf: Mapping[str, Any] | None,
    model: str,
    *,
    role: str | None = None,
) -> dict[str, Any]:
    """Merge global llama.cpp config with role/model specific overrides."""

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
