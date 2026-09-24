"""Staged analysis pipeline for Podcast Reels Forge.

This module owns the multi-stage local-only analysis flow:
scout -> cleanup -> refine -> judge -> metadata.

The root script `podcast_reels_forge/scripts/analyze.py` re-exports the public
helpers here so CLI entrypoints stay thin while tests can still import the
historic helper functions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable: object, **_: object) -> object:
        return iterable

from podcast_reels_forge.analysis.candidate_extraction import (
    build_candidate_json,
    normalize_candidate_list,
)
from podcast_reels_forge.analysis.audio_features import (
    annotate_records_with_audio,
    resolve_source_audio,
)
from podcast_reels_forge.analysis.chunking import (
    adaptive_overlap_seconds,
    build_analysis_chunks,
    estimate_tokens,
)
from podcast_reels_forge.analysis.contracts import (
    MomentRecord,
    coerce_moment_record,
    has_quote_evidence,
    replace_record,
)
from podcast_reels_forge.analysis.decisions import (
    CLEANUP_EDITABLE,
    JUDGE_EDITABLE,
    apply_stage_decisions,
)
from podcast_reels_forge.analysis.metadata import finalize_moment_list
from podcast_reels_forge.analysis.ranking import (
    DEFAULT_MAX_OVERLAP_RATIO,
    DEFAULT_MMR_LAMBDA,
    assign_clip_types,
    dedupe_moments,
    overlap_ratio_of_shorter,
    rank_moments,
    ranking_value,
    topic_similarity,
)
from podcast_reels_forge.analysis.scoring import clip_type_target_bounds
from podcast_reels_forge.analysis.serializers import atomic_write_json
from podcast_reels_forge.analysis.transcript_index import TranscriptIndex
from podcast_reels_forge.analysis.validation import (
    METHOD_EXACT,
    annotate_speech_rate,
    apply_quote_verification,
    clamp_record_to_window,
    clamp_records_to_episode,
    enforce_quote_containment,
    filter_nonoverlapping_outputs,
    quote_verification_settings,
    snap_records,
    split_by_quote_ratio,
)

from podcast_reels_forge.config import (
    LlamaCppRoleMapping,
    merge_llama_cpp_role_conf,
    resolve_llama_cpp_role_mapping,
)
from podcast_reels_forge.llm.providers import (
    AnthropicConfig,
    AnthropicProvider,
    GeminiConfig,
    GeminiProvider,
    LLMProvider,
    LlamaCppConfig,
    LlamaCppProvider,
    OpenAIConfig,
    OpenAIProvider,
    close_provider,
)
from podcast_reels_forge.llm.schemas import (
    CLEANUP_DECISIONS_SCHEMA,
    EPISODE_CONTEXT_SCHEMA,
    FALLBACK_SCHEMAS,
    JUDGE_REVIEWS_SCHEMA,
    SCOUT_JSON_SCHEMA,
)
from podcast_reels_forge.utils.json_utils import extract_first_json_value
from podcast_reels_forge.utils.logging_utils import setup_logging
from podcast_reels_forge.utils.llama_cpp_service import (
    ENV_MANAGED_BY_PIPELINE,

    llama_cpp_start,
    llama_cpp_stop,

    parse_local_llama_cpp_host_port,
)
from podcast_reels_forge.utils.reel_markdown import (
    build_description_text,
    build_hashtags,
)


LOGGER = setup_logging()

Moment = MomentRecord

# Candidates per cleanup request. Sized for the OUTPUT as much as the input:
# the stage echoes surviving records back, and grammar-constrained Cyrillic is
# token-hungry, so 25-record batches overflowed n_predict=4096 inside
# ctx_size=8192 and lost the tail of the JSON (seen on a real run).
_CLEANUP_CAP = 16

_STAGE_FILES = {
    "scout": "chunk",
    "cleanup_refine": "cleanup",
    "judge_metadata": "judge",
}

_LEGACY_STAGE_FALLBACKS = {
    "scout": ("chunk", "select"),
    "cleanup": ("select", "chunk"),
    "refine": ("select", "chunk"),
    "judge": ("select", "chunk"),
    "metadata": ("select", "chunk"),
}


def _status(msg: str, *, quiet: bool) -> None:
    if not quiet:
        LOGGER.info(msg)


def fmt_hms(sec: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""

    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def segments_to_compact_text(segments: list[dict[str, Any]], max_chars: int) -> str:
    """Convert transcript segments into a compact prompt-friendly text block."""

    lines: list[str] = []
    for seg in segments:
        try:
            start = int(float(seg.get("start", 0)))
            end = int(float(seg.get("end", 0)))
        except (TypeError, ValueError):
            start, end = 0, 0
        text = str(seg.get("text", "")).strip()

        if text:
            speaker = str(seg.get("speaker", "")).strip()
            prefix = f"({speaker}) " if speaker else ""
            lines.append(f"[{start}-{end}] {prefix}{text}")

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars]
        last_newline = result.rfind("\n")
        if last_newline > max_chars * 0.8:
            result = result[:last_newline]
    return result


def chunk_segments_by_time(
    segments: list[dict[str, Any]],
    chunk_seconds: int,
) -> list[list[dict[str, Any]]]:
    """Backward-compatible time chunking helper used by tests and legacy code."""

    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    chunk_start = float(segments[0]["start"]) if segments else 0.0
    for s in segments:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        if end - chunk_start <= float(chunk_seconds):
            cur.append(s)
        else:
            if cur:
                chunks.append(cur)
            cur = [s]
            chunk_start = start
    if cur:
        chunks.append(cur)
    return chunks


def _render_prompt(template: str, values: dict[str, str]) -> str:
    """Render a prompt template without treating braces as format fields."""

    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def _prompt_variant_for_model(prompts_conf: Mapping[str, Any], model: str) -> str:
    variant = str(prompts_conf.get("variant", "default"))
    mv = prompts_conf.get("model_variants")
    if isinstance(mv, Mapping):
        mvv = mv.get(model)
        if isinstance(mvv, str) and mvv.strip():
            return mvv.strip()
    return variant


def _normalize_prompt_lang(prompt_lang: str | None, transcript_lang: str | None) -> str:
    pl = (prompt_lang or "auto").strip().lower()
    if pl != "auto":
        return pl
    tl = (transcript_lang or "").strip().lower()
    if tl.startswith("ru"):
        return "ru"
    if tl.startswith("en"):
        return "en"
    return "ru"


def _load_prompt(*, lang: str, variant: str, name: str) -> str:
    """Load a prompt file with sensible fallbacks."""

    repo_prompts = Path(__file__).resolve().parent.parent.parent / "prompts"
    base = repo_prompts / lang
    candidates = [
        base / f"{name}_{variant}.txt",
        base / f"{name}_default.txt",
    ]
    for legacy_name in _LEGACY_STAGE_FALLBACKS.get(name, ()):
        candidates.extend(
            [
                base / f"{legacy_name}_{variant}.txt",
                base / f"{legacy_name}_default.txt",
            ],
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt template not found for stage '{name}' in {base}")


def _load_diarization(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []

    diar_path = Path(path)
    if not diar_path.exists():
        return []

    try:
        data = json.loads(diar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []

    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def _assign_speakers(
    segments: list[dict[str, Any]],
    diar: list[dict[str, Any]],
    *,
    prefix: bool = False,
) -> None:
    if not diar:
        return

    def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    for segment in segments:
        try:
            s0 = float(segment.get("start", 0))
            s1 = float(segment.get("end", 0))
        except (TypeError, ValueError):
            continue

        best_spk = None
        best_ov = 0.0
        for diar_entry in diar:
            try:
                d0 = float(diar_entry.get("start", 0))
                d1 = float(diar_entry.get("end", 0))
                spk = str(diar_entry.get("speaker", ""))
            except (TypeError, ValueError):
                continue
            ov = overlap(s0, s1, d0, d1)
            if ov > best_ov and spk:
                best_ov = ov
                best_spk = spk

        if best_spk:
            segment["speaker"] = best_spk
            if (
                prefix
                and isinstance(segment.get("text"), str)
                and not segment["text"].lstrip().startswith("(")
            ):
                segment["text"] = f"({best_spk}) {segment['text']}"


def _read_json_if_valid(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _coerce_json_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("moments", "candidates", "results", "items", "clips"):
            raw = payload.get(key)
            if isinstance(raw, list):
                return [item for item in raw if isinstance(item, dict)]
    return []


_JSON_RETRY_NOTE = (
    "\n\nВАЖНО: предыдущий ответ не удалось разобрать как JSON. "
    "Верни ТОЛЬКО валидный JSON по схеме выше, без пояснений и markdown.\n"
    "IMPORTANT: the previous answer could not be parsed as JSON. "
    "Return ONLY valid JSON matching the schema above, no prose, no markdown."
)


class CachingProvider:
    """RU: Дисковый кэш ответов LLM по хэшу запроса.

    EN: On-disk cache of LLM answers keyed by a hash of the request.

    The analysis is the one long LLM stage that dies halfway most often (a
    stalled server, an OOM, a reboot); with this cache a re-run replays every
    answer it already has and only pays for the rest. The key covers the
    model, the schema, the sampling temperature and the exact prompt, so any
    change to them is a cache miss. Entries not used by a run are pruned at
    its end (see :meth:`prune`).
    """

    def __init__(
        self,
        inner: LLMProvider,
        cache_dir: Path,
        *,
        namespace: str,
        used: set[str],
    ) -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        self.namespace = namespace
        self.used = used
        self.hits = 0

    def _key(self, prompt: str, temperature: float) -> str:
        material = json.dumps([self.namespace, round(float(temperature), 4), prompt], ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        key = self._key(prompt, temperature)
        self.used.add(key)
        path = self.cache_dir / f"{key}.json"
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and isinstance(cached.get("text"), str):
                self.hits += 1
                return cached["text"]
        except (OSError, ValueError):
            pass
        text = await self.inner.generate(prompt, temperature=temperature, timeout=timeout)
        try:
            atomic_write_json(path, {"text": text})
        except OSError as exc:
            LOGGER.debug("llm cache write failed: %s", exc)
        return text

    async def aclose(self) -> None:
        await close_provider(self.inner)

    @staticmethod
    def prune(cache_dir: Path, used: set[str]) -> int:
        removed = 0
        if not cache_dir.is_dir():
            return 0
        for path in cache_dir.glob("*.json"):
            if path.stem not in used:
                path.unlink(missing_ok=True)
                removed += 1
        return removed


class RetryBudget:
    """RU: Общий на эпизод лимит повторных запросов из-за битого JSON.

    EN: Episode-wide cap on re-asks after unparseable JSON. Without it a
    model that keeps misbehaving on a long episode turns every chunk into
    extra full-size requests — a retry avalanche on an already slow server.
    """

    def __init__(self, total: int) -> None:
        self.total = max(0, int(total))
        self.used = 0
        self.refused = 0

    def take(self) -> bool:
        if self.used >= self.total:
            self.refused += 1
            return False
        self.used += 1
        return True


async def get_llm_json(
    provider: LLMProvider,
    prompt: str,
    temperature: float,
    timeout: int,
    *,
    retries: int = 0,
    budget: RetryBudget | None = None,
) -> dict[str, Any] | list[Any]:
    """Get JSON from an LLM response, logging a safe preview on failure.

    An unparseable answer costs a whole chunk, so the request is re-issued up
    to ``retries`` times with an explicit note about the malformed output —
    as long as the episode-wide ``budget`` allows it. Truncated JSON is not a
    reason to retry: the salvage parser already recovers every complete item.
    """

    attempts = max(1, 1 + int(retries))
    for attempt in range(1, attempts + 1):
        prompt_text = prompt if attempt == 1 else prompt + _JSON_RETRY_NOTE
        raw = await provider.generate(prompt_text, temperature=temperature, timeout=timeout)
        try:
            return extract_first_json_value(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            preview = raw[:500].replace("\n", "\\n") if isinstance(raw, str) else str(raw)
            if attempt < attempts and (budget is None or budget.take()):
                LOGGER.warning(
                    "Failed to parse JSON from LLM output (attempt %d/%d), retrying "
                    "(raw preview: %s)",
                    attempt,
                    attempts,
                    preview,
                )
                continue
            LOGGER.warning(
                "Failed to parse JSON from LLM output after %d attempt(s)%s; "
                "returning [] (raw preview: %s)",
                attempt,
                "" if attempt == attempts else " (episode retry budget spent)",
                preview,
            )
            break
    return []


def create_provider(
    provider_name: str,
    *,
    model: str,
    url: str | None = None,
    api_key: str | None = None,
    llama_cpp_fallback_models: list[str] | None = None,
    llama_cpp_watchdog: bool = True,
    llama_cpp_first_token_timeout_s: int = 120,
    llama_cpp_stall_timeout_s: int = 120,
    llama_cpp_log_interval_s: int = 10,
    llama_cpp_max_retries: int = 2,
    llama_cpp_n_predict: int = 4096,
    llama_cpp_json_schema: Mapping[str, Any] | None = None,
    llama_cpp_fallback_schema: Mapping[str, Any] | None = None,
    llama_cpp_retry_base_delay_s: float = 2.0,
    llama_cpp_retry_max_delay_s: float = 30.0,
) -> LLMProvider:
    """Create an LLM provider.

    Note: the cloud provider branches are legacy compatibility paths only and
    are not used by the default workflow.
    """

    if provider_name == "llama_cpp":
        return LlamaCppProvider(
            LlamaCppConfig(
                url=url or "http://127.0.0.1:11440/completion",
                model=model,
                watchdog_enabled=bool(llama_cpp_watchdog),
                first_token_timeout_s=int(llama_cpp_first_token_timeout_s),
                stall_timeout_s=int(llama_cpp_stall_timeout_s),
                log_interval_s=int(llama_cpp_log_interval_s),
                max_retries=int(llama_cpp_max_retries),
                n_predict=int(llama_cpp_n_predict),
                json_schema=llama_cpp_json_schema,
                fallback_schema=llama_cpp_fallback_schema,
                retry_base_delay_s=float(llama_cpp_retry_base_delay_s),
                retry_max_delay_s=float(llama_cpp_retry_max_delay_s),
                fallback_models=tuple(llama_cpp_fallback_models or []),
            ),
        )
    if provider_name == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Missing OpenAI key. Set OPENAI_API_KEY or pass --api-key")
        return OpenAIProvider(OpenAIConfig(api_key=key, model=model))
    if provider_name == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(
                "Missing Anthropic key. Set ANTHROPIC_API_KEY or pass --api-key",
            )
        return AnthropicProvider(AnthropicConfig(api_key=key, model=model))
    if provider_name == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Missing Gemini key. Set GEMINI_API_KEY or pass --api-key")
        return GeminiProvider(GeminiConfig(api_key=key, model=model))
    raise SystemExit(f"Unsupported provider: {provider_name}")


def _provider_for_role(
    provider_name: str,
    model: str,
    *,
    base_url: str | None,
    role_conf: Mapping[str, Any] | None,
    api_key: str | None,
    json_schema: Mapping[str, Any] | None = None,
    fallback_schema: Mapping[str, Any] | None = None,
) -> LLMProvider:
    conf = dict(role_conf or {})
    watchdog = conf.get("watchdog", {})
    if not isinstance(watchdog, Mapping):
        watchdog = {}
    return create_provider(
        provider_name,
        model=model,
        url=base_url if provider_name == "llama_cpp" else None,
        api_key=api_key,
        llama_cpp_fallback_models=[
            str(item).strip()
            for item in conf.get("fallback_models", [])
            if str(item).strip()
        ],
        llama_cpp_watchdog=bool(watchdog.get("enabled", conf.get("watchdog_enabled", True))),
        llama_cpp_first_token_timeout_s=int(
            watchdog.get("first_token_timeout", conf.get("first_token_timeout_s", 120)),
        ),
        llama_cpp_stall_timeout_s=int(
            watchdog.get("stall_timeout", conf.get("stall_timeout_s", 120)),
        ),
        llama_cpp_log_interval_s=int(
            watchdog.get("log_interval", conf.get("log_interval_s", 10)),
        ),
        llama_cpp_max_retries=int(
            watchdog.get("max_retries", conf.get("max_retries", 2)),
        ),
        llama_cpp_n_predict=int(conf.get("n_predict", 4096)),
        llama_cpp_json_schema=json_schema,
        llama_cpp_fallback_schema=fallback_schema,
        llama_cpp_retry_base_delay_s=_conf_float(conf, "retry_base_delay_s", 2.0),
        llama_cpp_retry_max_delay_s=_conf_float(conf, "retry_max_delay_s", 30.0),
    )


def _stage_config(
    base_conf: Mapping[str, Any],
    *,
    role: str,
    model: str,
) -> dict[str, Any]:
    merged = merge_llama_cpp_role_conf(base_conf, model, role=role)
    return merged


def analysis_conf_section(
    processing_conf: Mapping[str, Any],
    *keys: str,
) -> Mapping[str, Any]:
    """RU: Достаёт вложенную секцию processing.analysis.*, терпя мусор.

    EN: Read a nested ``processing.analysis.*`` section, tolerating configs
    where the key is absent or holds the wrong type. Every knob has a code
    default, so the whole block is optional.
    """

    section: Mapping[str, Any] = processing_conf
    for key in ("analysis", *keys):
        value = section.get(key)
        if not isinstance(value, Mapping):
            return {}
        section = value
    return section


def _conf_int(conf: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(conf[key])
    except (KeyError, TypeError, ValueError):
        return default


def _conf_float(conf: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(conf[key])
    except (KeyError, TypeError, ValueError):
        return default


def _conf_bool(conf: Mapping[str, Any], key: str, default: bool) -> bool:
    value = conf.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return default


def _stage_temperature(stage_conf: Mapping[str, Any], default: float) -> float:
    try:
        return float(stage_conf.get("temperature", default))
    except (TypeError, ValueError):
        return default


def _stage_timeout(stage_conf: Mapping[str, Any], default: int) -> int:
    try:
        return int(stage_conf.get("timeout", default))
    except (TypeError, ValueError):
        return default


def _stage_parallelism(stage_conf: Mapping[str, Any], default: int) -> int:
    try:
        return max(1, int(stage_conf.get("parallelism", default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def _stage_chunk_seconds(stage_conf: Mapping[str, Any], default: int) -> int:
    try:
        return int(stage_conf.get("chunk_seconds", default))
    except (TypeError, ValueError):
        return default


def _stage_max_chars(stage_conf: Mapping[str, Any], default: int) -> int:
    try:
        return int(stage_conf.get("max_chars_chunk", default))
    except (TypeError, ValueError):
        return default


def scale_quotas_to_duration(
    quotas: Mapping[str, int],
    *,
    duration_s: float,
    clips_per_hour: float,
) -> dict[str, int]:
    """RU: Масштабирует квоты клипов под хронометраж эпизода.

    Цель — ``clips_per_hour`` клипов на час общей длительности (считается от
    суммарного времени, не по часовым корзинам). Пропорции типов из конфига
    сохраняются; распределение — методом наибольших остатков, чтобы сумма
    сошлась точно.

    EN: Scale clip quotas to the episode duration. The target is
    ``clips_per_hour`` clips per hour of total runtime (computed from the
    total, not per-hour buckets). The configured type mix is preserved;
    apportionment uses the largest-remainder method so the counts sum exactly
    to the target.
    """

    if clips_per_hour <= 0 or duration_s <= 0:
        return dict(quotas)

    base = {key: max(0, int(value)) for key, value in quotas.items()}
    base_total = sum(base.values())
    target_total = max(1, round(duration_s / 3600.0 * clips_per_hour))
    if base_total <= 0:
        return {**base, "reel": target_total}

    shares = {
        key: value / base_total * target_total for key, value in base.items()
    }
    scaled = {key: int(share) for key, share in shares.items()}
    shortfall = target_total - sum(scaled.values())
    # Hand the remaining slots to the largest fractional parts.
    for key, _remainder in sorted(
        ((key, shares[key] - scaled[key]) for key in shares),
        key=lambda item: -item[1],
    )[:shortfall]:
        scaled[key] += 1
    return scaled


def _build_requirements_text(
    processing_conf: Mapping[str, Any],
    *,
    quotas: Mapping[str, int] | None = None,
    include_counts: bool = True,
) -> str:
    """The clip ask, as told to the model.

    When ``quotas`` is given (already scaled to the episode duration), the
    counts come from it, so the prompt and the selection enforce the same
    numbers; the config supplies only the per-type duration limits.

    ``include_counts=False`` renders the lengths only. That is what the scout
    gets: its job is recall inside one chunk, and an episode-wide quota
    there only made it pad or hold back; quotas are enforced once, by the
    final selector.
    """

    clips_conf = processing_conf.get("clips", {})
    if not isinstance(clips_conf, Mapping):
        clips_conf = {}

    def _count(bucket: str, conf_key: str, count_key: str = "count") -> int | None:
        if quotas is not None:
            return int(quotas.get(bucket, 0))
        section = clips_conf.get(conf_key)
        if isinstance(section, Mapping):
            return int(section.get(count_key, 0))
        return None

    def _max_duration(conf_key: str, default: int) -> int:
        section = clips_conf.get(conf_key)
        if isinstance(section, Mapping):
            try:
                return int(section.get("max_duration", default))
            except (TypeError, ValueError):
                return default
        return default

    def _clips(count: int) -> str:
        return f"{count} clips " if include_counts else ""

    parts: list[str] = []
    stories = _count("story", "stories")
    if stories is not None and (quotas is None or stories > 0):
        parts.append(f"Stories: {_clips(stories)}up to {_max_duration('stories', 15)}s")
    reels = _count("reel", "reels")
    if reels is not None and (quotas is None or reels > 0):
        parts.append(f"Reels: {_clips(reels)}up to {_max_duration('reels', 60)}s")
    long_reels = _count("long_reel", "long_reels")
    if long_reels is not None and (quotas is None or long_reels > 0):
        parts.append(
            f"Long reels: {_clips(long_reels)}up to {_max_duration('long_reels', 180)}s",
        )
    highlights = _count("highlight", "highlights", "moments_count")
    if highlights is not None and (quotas is None or highlights > 0):
        parts.append(
            f"Highlights: {highlights} moments" if include_counts else "Highlights: 10-30s",
        )

    if not parts:
        reel_min = processing_conf.get("reel_min_duration", 30)
        reel_max = processing_conf.get("reel_max_duration", 60)
        total = sum(quotas.values()) if quotas else 4
        parts.append(f"Reels: {_clips(max(1, total))}of {reel_min}-{reel_max}s")
    return "\n".join(parts)


def target_candidate_range(chunk_seconds: float) -> str:
    """How many scout candidates a chunk of this length usually holds.

    Roughly 4-8 per ten minutes. Without a range the model either returned
    too few out of caution or padded the list with noise.
    """

    minutes = max(0.0, float(chunk_seconds)) / 60.0
    low = max(1, round(minutes * 0.4))
    high = max(low + 2, round(minutes * 0.8))
    return f"{low}-{high}"


def _requested_quotas(processing_conf: Mapping[str, Any]) -> dict[str, int]:
    clips_conf = processing_conf.get("clips", {})
    if not isinstance(clips_conf, Mapping):
        clips_conf = {}

    quotas = {
        "story": int(clips_conf.get("stories", {}).get("count", 0))
        if isinstance(clips_conf.get("stories"), Mapping)
        else 0,
        "reel": int(clips_conf.get("reels", {}).get("count", 0))
        if isinstance(clips_conf.get("reels"), Mapping)
        else int(processing_conf.get("reels_count", 4)),
        "long_reel": int(clips_conf.get("long_reels", {}).get("count", 0))
        if isinstance(clips_conf.get("long_reels"), Mapping)
        else 0,
        "highlight": int(clips_conf.get("highlights", {}).get("moments_count", 0))
        if isinstance(clips_conf.get("highlights"), Mapping)
        else 0,
    }
    if quotas["reel"] <= 0:
        quotas["reel"] = int(processing_conf.get("reels_count", 4))
    return quotas


def render_reels_summary_markdown(moments: list[Moment]) -> str:
    """Render a compact markdown summary for final moments."""

    lines = ["# Reels Suggestions", ""]
    for i, m in enumerate(moments, 1):
        moment_data = m.to_dict() if isinstance(m, MomentRecord) else asdict(m)
        description = build_description_text(moment_data)
        hashtags = build_hashtags(moment_data, description_text=description)

        title = str(moment_data.get("title", "")).strip()
        lines.append(f"## {i}. {title} [{moment_data.get('clip_type', 'reel')}]")
        lines.append(f"Time: {fmt_hms(float(moment_data.get('start', 0)))}-{fmt_hms(float(moment_data.get('end', 0)))}")
        score_line = f"Score: {float(moment_data.get('score', 0)):.1f}/10"
        priority = moment_data.get("priority")
        if priority is not None:
            score_line += f" (priority {float(priority):.2f})"
        lines.append(score_line)
        why = str(moment_data.get("why", "")).strip()
        if why:
            lines.append(f"Why: {why}")
        hook = str(moment_data.get("hook", "")).strip()
        if hook:
            lines.append(f"Hook: {hook}")
        lines.append("")
        if description:
            lines.append(description)
            lines.append("")
        if hashtags:
            lines.append(" ".join(hashtags))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _attach_chunk_metadata(
    record: MomentRecord,
    *,
    chunk_id: str,
    speaker_set: Sequence[str],
    candidate_id: str = "",
) -> MomentRecord:
    payload = {
        **record.to_dict(),
        "source_chunk_ids": list(dict.fromkeys([*record.source_chunk_ids, chunk_id])),
    }
    if candidate_id:
        # Python owns identity: whatever id the model made up is replaced.
        payload["candidate_id"] = candidate_id
    if speaker_set and not payload.get("speaker"):
        payload["speaker"] = speaker_set[0]
        payload["speaker_confidence"] = 0.5 if len(speaker_set) == 1 else 0.35
    coerced = coerce_moment_record(payload)
    return coerced or record


def _parse_candidate_response(
    value: dict[str, Any] | list[Any],
    *,
    stage: str,
) -> list[MomentRecord]:
    candidates = normalize_candidate_list(value, stage=stage)
    return candidates


def _prompt_payload(
    *,
    requirements: str,
    chunk: Mapping[str, Any] | None = None,
    candidates: Sequence[MomentRecord] | None = None,
    transcript: str | None = None,
    episode_context: str = "",
    candidates_payload: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    # Always supply the key so an unused {episode_context} placeholder does not
    # survive into the rendered prompt.
    payload: dict[str, str] = {
        "requirements": requirements,
        "episode_context": episode_context,
    }
    if chunk is not None:
        try:
            span = float(chunk.get("end", 0.0)) - float(chunk.get("start", 0.0))
        except (TypeError, ValueError):
            span = 0.0
        payload["target_candidates"] = target_candidate_range(span)
        # Send the chunk text exactly once. The prompt carries the transcript in
        # its own {transcript} section, so keep only metadata/timecodes in
        # {chunk_json}; duplicating the text here doubled the scout input and,
        # at ctx=8192, forced llama.cpp to left-truncate the prompt (the model
        # lost the start of each chunk) while the output still hit n_predict.
        chunk_meta = {key: value for key, value in chunk.items() if key != "text"}
        payload["chunk_json"] = json.dumps(chunk_meta, ensure_ascii=False)
        payload["transcript"] = str(chunk.get("text", ""))
    if transcript is not None:
        payload["transcript"] = transcript
    if candidates_payload is not None:
        payload["candidates_json"] = json.dumps(
            list(candidates_payload),
            ensure_ascii=False,
        )
    elif candidates is not None:
        payload["candidates_json"] = json.dumps(
            build_candidate_json(candidates),
            ensure_ascii=False,
        )
    return payload


def _make_stage_provider(
    provider_name: str,
    *,
    model: str,
    base_url: str | None,
    stage_conf: Mapping[str, Any],
    api_key: str | None,
    json_schema: Mapping[str, Any] | None = None,
    fallback_schema: Mapping[str, Any] | None = None,
) -> LLMProvider:
    return _provider_for_role(
        provider_name,
        model,
        base_url=base_url,
        role_conf=stage_conf,
        api_key=api_key,
        json_schema=json_schema,
        fallback_schema=fallback_schema,
    )


def _default_stage_temperature(role: str) -> float:
    if role == "scout":
        return 0.35
    if role == "cleanup_refine":
        return 0.15
    if role == "judge_metadata":
        return 0.05
    return 0.15


def _stage_name_to_prompt_name(stage: str) -> str:
    return _STAGE_FILES.get(stage, stage)


@dataclass
class StageStats:
    """RU: Счётчики и задержки одной LLM-стадии для analysis_metrics.json.

    EN: Counters and latency of one LLM stage, for analysis_metrics.json.
    """

    calls: int = 0
    failed_calls: int = 0
    seconds: float = 0.0
    prompt_chars: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, amount: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(amount)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "seconds": round(self.seconds, 2),
            "prompt_chars": self.prompt_chars,
            **self.counts,
        }


async def _timed_llm_json(
    provider: LLMProvider,
    prompt_text: str,
    temperature: float,
    timeout: int,
    *,
    retries: int,
    budget: RetryBudget | None,
    stats: StageStats | None,
) -> dict[str, Any] | list[Any]:
    started = time.monotonic()
    try:
        return await get_llm_json(
            provider, prompt_text, temperature, timeout, retries=retries, budget=budget,
        )
    except Exception:
        if stats is not None:
            stats.failed_calls += 1
        raise
    finally:
        if stats is not None:
            stats.calls += 1
            stats.seconds += time.monotonic() - started
            stats.prompt_chars += len(prompt_text)


async def scout_candidates(
    provider: LLMProvider,
    chunks: Sequence[Any],
    *,
    requirements: str,
    prompt: str,
    temperature: float,
    timeout: int,
    progress: bool = False,
    parallelism: int = 1,
    json_retries: int = 0,
    chunk_tolerance_s: float = 3.0,
    episode_context: str = "",
    budget: RetryBudget | None = None,
    stats: StageStats | None = None,
) -> list[MomentRecord]:
    """Scout each chunk for candidates.

    Every surviving candidate gets a stable ``candidate_id``
    (``<chunk_id>_cNN``) assigned here, in Python: cleanup and judge refer to
    candidates by it. A candidate without a quote to verify is dropped on
    the spot — it can never be proven.

    A chunk that fails outright (the provider exhausted its retries, the
    server went away) is logged and skipped: losing one 20-minute window is a
    far better outcome than losing the whole episode. If every chunk fails the
    caller is told, since that is a real outage rather than a bad chunk.
    """

    candidates: list[MomentRecord] = []
    chunk_list = list(chunks)
    if not chunk_list:
        return candidates

    sem = asyncio.Semaphore(parallelism)
    total = len(chunk_list)
    failures = 0

    async def _process_chunk(index: int, chunk: Any) -> list[MomentRecord]:
        nonlocal failures
        async with sem:
            chunk_id = getattr(chunk, "chunk_id", f"chunk_{index:03d}")
            try:
                prompt_text = _render_prompt(
                    prompt,
                    _prompt_payload(
                        requirements=requirements,
                        chunk=chunk.to_prompt_dict() if hasattr(chunk, "to_prompt_dict") else None,
                        episode_context=episode_context,
                    ),
                )
                resp = await _timed_llm_json(
                    provider, prompt_text, temperature, timeout,
                    retries=json_retries, budget=budget, stats=stats,
                )
            except Exception as exc:
                failures += 1
                LOGGER.warning(
                    "scout failed on %s (%d/%d); skipping this chunk: %s",
                    chunk_id, index, total, exc,
                )
                return []

            chunk_candidates = _parse_candidate_response(resp, stage="scout")
            window_start = getattr(chunk, "start", None)
            window_end = getattr(chunk, "end", None)
            out: list[MomentRecord] = []
            outside = 0
            no_quote = 0
            for candidate in chunk_candidates:
                if not has_quote_evidence(candidate):
                    no_quote += 1
                    continue
                # Keep candidates inside the window the model was actually
                # shown; anything further out is a hallucinated timecode.
                if window_start is not None and window_end is not None:
                    clamped = clamp_record_to_window(
                        candidate,
                        float(window_start),
                        float(window_end),
                        tolerance_s=chunk_tolerance_s,
                    )
                    if clamped is None:
                        outside += 1
                        continue
                    candidate = clamped
                out.append(
                    _attach_chunk_metadata(
                        candidate,
                        chunk_id=chunk_id,
                        speaker_set=getattr(chunk, "speaker_set", ()),
                        candidate_id=f"{chunk_id}_c{len(out) + 1:02d}",
                    ),
                )
            if stats is not None:
                stats.bump("returned", len(chunk_candidates))
                stats.bump("dropped_outside_window", outside)
                stats.bump("dropped_without_quote", no_quote)
            if outside or no_quote:
                LOGGER.info(
                    "%s: dropped %d candidate(s) outside the chunk window, "
                    "%d without a quote",
                    chunk_id, outside, no_quote,
                )
            if progress:
                LOGGER.info(
                    "[scout] %s (%d/%d): %d candidates", chunk_id, index, total, len(out),
                )
            return out

    tasks = [_process_chunk(i, c) for i, c in enumerate(chunk_list, 1)]
    results = await asyncio.gather(*tasks)

    if failures == total:
        raise RuntimeError(
            f"scout failed on all {total} chunk(s); the llama.cpp server is likely down",
        )
    if failures:
        LOGGER.warning("scout skipped %d of %d chunk(s) after errors", failures, total)

    for res in results:
        candidates.extend(res)
    return candidates


def build_cleanup_payload(records: Sequence[MomentRecord]) -> list[dict[str, Any]]:
    """What cleanup is shown: identity, interval and evidence — nothing else.

    Titles, captions and scores of other stages would only invite the model
    to echo or rewrite them; the decision needs the quote and the evidence.
    """

    payload: list[dict[str, Any]] = []
    for record in records:
        item: dict[str, Any] = {
            "candidate_id": record.candidate_id,
            "start": round(record.start, 1),
            "end": round(record.end, 1),
            "quote": record.quote,
            "evidence": record.why,
            "score": record.score,
        }
        if record.reason_codes:
            item["reason_codes"] = list(record.reason_codes)
        if record.quote_match_ratio is not None:
            item["quote_match_ratio"] = record.quote_match_ratio
        payload.append(item)
    return payload


def _ensure_candidate_ids(records: Sequence[MomentRecord], prefix: str) -> list[MomentRecord]:
    """Give id-less records (legacy callers, tests) a stable id."""

    out: list[MomentRecord] = []
    for offset, record in enumerate(records, 1):
        if record.candidate_id:
            out.append(record)
        else:
            out.append(replace_record(record, candidate_id=f"{prefix}_{offset:03d}"))
    return out


async def _gather_batches(
    batches: Sequence[Sequence[MomentRecord]],
    run_batch: Any,
    *,
    parallelism: int,
) -> list[list[MomentRecord]]:
    sem = asyncio.Semaphore(max(1, int(parallelism)))

    async def _guarded(batch: Sequence[MomentRecord]) -> list[MomentRecord]:
        async with sem:
            result: list[MomentRecord] = await run_batch(batch)
            return result

    return list(await asyncio.gather(*(_guarded(batch) for batch in batches)))


async def cleanup_and_refine_candidates(
    provider: LLMProvider,
    candidates: Sequence[MomentRecord],
    *,
    requirements: str,
    prompt: str,
    temperature: float,
    timeout: int,
    max_items: int,
    json_retries: int = 0,
    batch_size: int = _CLEANUP_CAP,
    parallelism: int = 1,
    budget: RetryBudget | None = None,
    stats: StageStats | None = None,
) -> list[MomentRecord]:
    """Filter and deduplicate candidates through keep/drop/merge decisions.

    The model answers with decisions by ``candidate_id``; they are applied to
    the original records, so quotes and timecodes cannot drift here. Batches
    are split by time order so overlapping near-duplicates land in the same
    call and can be merged; independent batches run in parallel (up to
    ``parallelism``), and the final dedupe catches strays across borders.
    """

    cleaned_input = _ensure_candidate_ids(dedupe_moments(candidates), "cand")
    if not cleaned_input:
        return []

    sorted_candidates = sorted(
        cleaned_input,
        key=lambda record: (-ranking_value(record), record.start, record.end),
    )[:max(1, int(max_items))]

    batch_size = max(1, int(batch_size))
    by_time = sorted(sorted_candidates, key=lambda record: (record.start, record.end))
    batches = [by_time[offset : offset + batch_size] for offset in range(0, len(by_time), batch_size)]

    async def _run(batch: Sequence[MomentRecord]) -> list[MomentRecord]:
        prompt_text = _render_prompt(
            prompt,
            _prompt_payload(
                requirements=requirements,
                candidates_payload=build_cleanup_payload(batch),
            ),
        )
        try:
            resp = await _timed_llm_json(
                provider, prompt_text, temperature, timeout,
                retries=json_retries, budget=budget, stats=stats,
            )
        except Exception as exc:
            # A failed batch keeps its input rather than dropping that
            # stretch of the episode.
            LOGGER.warning("cleanup batch failed; keeping its input: %s", exc)
            return list(batch)
        outcome = apply_stage_decisions(
            batch, resp, stage="cleanup_refine", editable=CLEANUP_EDITABLE,
        )
        if stats is not None:
            stats.bump("dropped", len(outcome.dropped))
            stats.bump("merged", len(outcome.merged))
            stats.bump("unmentioned", outcome.unmentioned)
            stats.bump("untraceable", outcome.untraceable)
        return outcome.records

    results = await _gather_batches(batches, _run, parallelism=parallelism)
    refined = [record for batch in results for record in batch]
    return dedupe_moments(refined) if refined else list(sorted_candidates)


def stratified_batches(
    records: Sequence[MomentRecord],
    batch_size: int,
) -> list[list[MomentRecord]]:
    """Split records into batches that each span the whole quality range.

    Priority-ordered slicing gave the first judge call the best candidates
    and the last one the weakest, so "9/10" meant different things in
    different calls. Dealing the ranked list round-robin gives every batch
    the same mix, which keeps the judge's absolute scores comparable across
    calls; the global comparison itself happens deterministically in
    :func:`rank_moments`.
    """

    ordered = sorted(records, key=ranking_value, reverse=True)
    if not ordered:
        return []
    count = max(1, math.ceil(len(ordered) / max(1, int(batch_size))))
    batches: list[list[MomentRecord]] = [[] for _ in range(count)]
    for offset, record in enumerate(ordered):
        batches[offset % count].append(record)
    return batches


async def judge_candidates(
    provider: LLMProvider,
    candidates: Sequence[MomentRecord],
    *,
    requirements: str,
    prompt: str,
    temperature: float,
    timeout: int,
    json_retries: int = 0,
    episode_context: str = "",
    candidates_payload: Sequence[Mapping[str, Any]] | None = None,
    budget: RetryBudget | None = None,
    stats: StageStats | None = None,
) -> list[MomentRecord]:
    """Review one batch of candidates.

    The judge answers with reviews by ``candidate_id``: keep or drop, a 1-10
    score and presentation metadata (title, hook, why). Quotes and bounds
    stay the source's. Ranking, quota selection and metadata finalization all
    happen once, in the caller.
    """

    if not candidates:
        return []

    candidates = _ensure_candidate_ids(candidates, "judge")
    prompt_text = _render_prompt(
        prompt,
        _prompt_payload(
            requirements=requirements,
            candidates=candidates,
            candidates_payload=candidates_payload,
            episode_context=episode_context,
        ),
    )
    resp = await _timed_llm_json(
        provider, prompt_text, temperature, timeout,
        retries=json_retries, budget=budget, stats=stats,
    )
    outcome = apply_stage_decisions(
        candidates,
        resp,
        stage="judge_metadata",
        editable=JUDGE_EDITABLE,
        record_judge_score=True,
    )
    if stats is not None:
        stats.bump("dropped", len(outcome.dropped))
        stats.bump("unmentioned", outcome.unmentioned)
        stats.bump("untraceable", outcome.untraceable)
    return outcome.records


_DIGEST_SIGNAL_RE = re.compile(r"\d|[?!？！]|\b(?:смех|laugh|haha|ха-ха)", re.IGNORECASE)


def build_transcript_digest(index: TranscriptIndex, *, max_chars: int = 4000) -> str:
    """RU: Многоканальная выжимка эпизода для обзора.

    EN: A multi-channel digest of the episode. Half the budget goes to an
    even sample (roughly one sentence per window, for coverage); the rest to
    sentences that carry signal — numbers, questions, exclamations,
    laughter — and to speaker changes, which is where climaxes, arguments and
    punchlines live and where an even sample tends to miss. Sentences are
    emitted in episode order with their timestamp.
    """

    if not index.sentences:
        return ""

    sentences = index.sentences
    episode_end = sentences[-1].end

    def _size(positions: set[int]) -> int:
        return sum(len(sentences[p].text) + 12 for p in positions)

    def _even_sample(window: float) -> set[int]:
        chosen: set[int] = set()
        next_slot = sentences[0].start
        for position, sentence in enumerate(sentences):
            if sentence.start < next_slot:
                continue
            chosen.add(position)
            next_slot = sentence.start + window
        return chosen

    # The even sample gets about half the budget; widen its window until it
    # fits, so coverage reaches the end of the episode instead of being cut.
    window = max(60.0, episode_end / 40.0)
    picked = _even_sample(window)
    while _size(picked) > max_chars * 0.5 and window < episode_end:
        window *= 1.5
        picked = _even_sample(window)

    signal = [
        position
        for position, sentence in enumerate(sentences)
        if position not in picked and _DIGEST_SIGNAL_RE.search(sentence.text)
    ]
    # Spread the signal picks across the episode rather than taking the first
    # ones: every k-th, until the budget is used.
    if signal:
        step = max(1, len(signal) // 20)
        for position in signal[::step]:
            if _size(picked) + len(sentences[position].text) > max_chars:
                break
            picked.add(position)

    lines = [
        f"[{fmt_hms(sentences[p].start)}] {sentences[p].text}"
        for p in sorted(picked)
    ]
    digest = "\n".join(lines).strip()
    if len(digest) > max_chars:
        digest = digest[:max_chars].rsplit(" ", 1)[0] + "…"
    return digest


def format_episode_context(payload: Mapping[str, Any]) -> str:
    """Render the episode overview as a prompt section."""

    summary = str(payload.get("summary", "")).strip()
    topics = payload.get("topics")
    tone = str(payload.get("tone", "")).strip()
    speakers = payload.get("speakers")
    limits = payload.get("context_limits")

    def _joined(values: Any) -> str:
        if not isinstance(values, list):
            return ""
        return ", ".join(str(value).strip() for value in values if str(value).strip())

    lines: list[str] = []
    if summary:
        lines.append(summary)
    if _joined(topics):
        lines.append(f"Темы эпизода / Episode topics: {_joined(topics)}")
    if tone:
        lines.append(f"Тональность / Tone: {tone}")
    if _joined(speakers):
        lines.append(f"Участники / Speakers: {_joined(speakers)}")
    if _joined(limits):
        lines.append(f"Клипу нужно пояснить / A clip must explain: {_joined(limits)}")

    if not lines:
        return ""
    return "# Контекст эпизода / Episode context\n" + "\n".join(lines)


# Bump when the context payload or its schema changes shape.
_EPISODE_CONTEXT_SCHEMA_VERSION = 2


def episode_context_cache_key(
    *,
    digest: str,
    prompt: str,
    model: str,
    lang: str,
) -> str:
    """Hash of everything the cached overview depends on."""

    material = json.dumps(
        {
            "digest": digest,
            "prompt": prompt,
            "model": model,
            "lang": lang,
            "schema": _EPISODE_CONTEXT_SCHEMA_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


async def build_episode_context(
    provider: LLMProvider,
    index: TranscriptIndex,
    *,
    outdir: Path,
    lang: str,
    variant: str,
    temperature: float,
    timeout: int,
    max_digest_chars: int = 4000,
    json_retries: int = 0,
    model: str = "",
    budget: RetryBudget | None = None,
    stats: StageStats | None = None,
) -> str:
    """Summarize the episode once, so the scout can judge moments in context.

    A moment can look striking inside its own chunk and be unremarkable for
    the episode; the scout has no way to tell without this. The result is
    cached in ``episode_context.json`` under a key built from the digest
    (i.e. the transcript), the prompt, the model and the language, so a
    changed input is never answered from a stale cache. Entirely
    best-effort: any failure returns an empty string and the prompts render
    without the section.
    """

    digest = build_transcript_digest(index, max_chars=max_digest_chars)
    if not digest:
        return ""

    try:
        prompt = _load_prompt(lang=lang, variant=variant, name="context")
    except FileNotFoundError:
        LOGGER.info("no episode-context prompt for lang=%s; skipping", lang)
        return ""

    cache_key = episode_context_cache_key(digest=digest, prompt=prompt, model=model, lang=lang)
    cache_path = outdir / "episode_context.json"
    cached = _read_json_if_valid(cache_path)
    if (
        isinstance(cached, dict)
        and cached.get("summary")
        and cached.get("cache_key") == cache_key
    ):
        return format_episode_context(cached)

    try:
        resp = await _timed_llm_json(
            provider,
            _render_prompt(prompt, {"transcript_digest": digest}),
            temperature,
            timeout,
            retries=json_retries,
            budget=budget,
            stats=stats,
        )
    except Exception as exc:
        LOGGER.warning("episode context failed; continuing without it: %s", exc)
        return ""

    if not isinstance(resp, dict) or not resp.get("summary"):
        LOGGER.info("episode context returned no summary; continuing without it")
        return ""

    atomic_write_json(cache_path, {**resp, "cache_key": cache_key, "model": model, "lang": lang})
    return format_episode_context(resp)


def build_judge_payload(
    records: Sequence[MomentRecord],
    index: TranscriptIndex,
    *,
    max_candidates: int = 14,
    head_seconds: float = 15.0,
    tail_seconds: float = 5.0,
    max_excerpt_chars: int = 260,
) -> list[dict[str, Any]]:
    """Candidate payload for the judge, including real opening/closing text.

    The judge is asked to reward strong first seconds and penalize ragged
    endings, which it cannot do from metadata alone — so give it the actual
    words at both ends of each clip. Kept small on purpose: this all has to
    fit inside ctx_size=8192 alongside the instructions.
    """

    ordered = sorted(records, key=ranking_value, reverse=True)[: max(1, int(max_candidates))]

    payload: list[dict[str, Any]] = []
    for record in ordered:
        item: dict[str, Any] = {
            "candidate_id": record.candidate_id,
            "start": round(record.start, 1),
            "end": round(record.end, 1),
            "duration": round(record.end - record.start, 1),
            "quote": record.quote,
            "evidence": record.why,
            "score": record.score,
        }
        if record.title:
            item["title"] = record.title
        if record.reason_codes:
            item["reason_codes"] = list(record.reason_codes)
        if record.speaker:
            item["speaker"] = record.speaker
        if record.quote_match_ratio is not None:
            item["quote_match_ratio"] = record.quote_match_ratio

        if index:
            head = index.text_between(
                record.start,
                min(record.start + head_seconds, record.end),
                max_chars=max_excerpt_chars,
            )
            tail = index.text_between(
                max(record.end - tail_seconds, record.start),
                record.end,
                max_chars=max_excerpt_chars // 2,
            )
            if head:
                item["excerpt_head"] = head
            if tail:
                item["excerpt_tail"] = tail
        payload.append(item)
    return payload


def _guard_stage_output(
    outputs: Sequence[MomentRecord],
    inputs: Sequence[MomentRecord],
    *,
    stage: str,
    enabled: bool,
) -> list[MomentRecord]:
    """Drop records a filtering stage invented rather than selected.

    With decision objects every output is built from an input, so this is a
    backstop for legacy answers and custom prompts. Falls back to the stage
    input if the guard would empty the list, so a misbehaving model costs
    precision rather than the whole episode.
    """

    if not enabled or not outputs:
        return list(outputs)

    kept = filter_nonoverlapping_outputs(outputs, inputs)
    if len(kept) == len(outputs):
        return kept
    if not kept:
        LOGGER.warning(
            "%s returned %d record(s), none traceable to its input; keeping the input",
            stage, len(outputs),
        )
        return list(inputs)
    LOGGER.warning(
        "%s: dropped %d record(s) that could not be traced to an input candidate",
        stage, len(outputs) - len(kept),
    )
    return kept


def _ensure_prompt_text(stage: str, lang: str, variant: str) -> str:
    return _load_prompt(lang=lang, variant=variant, name=_stage_name_to_prompt_name(stage))


def _rejection_rows(
    records: Sequence[MomentRecord],
    *,
    stage: str,
    reason: str,
) -> list[dict[str, Any]]:
    return [
        {**record.to_dict(), "rejected_at": stage, "rejection_reason": reason}
        for record in records
    ]


def _percentile(values: Sequence[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))
    return round(ordered[position], 3)


def _rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _topic_diversity(records: Sequence[MomentRecord]) -> float | None:
    """1 - mean pairwise topic similarity of the final set (1 = all distinct)."""

    pairs = [
        topic_similarity(first, second)
        for offset, first in enumerate(records)
        for second in records[offset + 1 :]
    ]
    if not pairs:
        return None
    return round(1.0 - sum(pairs) / len(pairs), 4)


def _warn_on_context_budget(
    chunks: Sequence[Any],
    *,
    prompt: str,
    llama_cpp_conf: Mapping[str, Any],
    n_predict: int,
) -> None:
    """Warn when the biggest scout prompt plus its output may overflow ctx_size.

    ``max_chars_chunk`` is a character budget, but the context is measured in
    tokens, and Cyrillic costs roughly half again as many per character. An
    overflow makes llama.cpp left-truncate the prompt silently.
    """

    service = llama_cpp_conf.get("service")
    ctx_size = _conf_int(service, "ctx_size", 0) if isinstance(service, Mapping) else 0
    if ctx_size <= 0 or not chunks:
        return
    longest = max(len(getattr(chunk, "text", "")) for chunk in chunks)
    sample = next(chunk for chunk in chunks if len(getattr(chunk, "text", "")) == longest)
    estimate = estimate_tokens(prompt) + estimate_tokens(getattr(sample, "text", ""))
    if estimate + n_predict > ctx_size:
        LOGGER.warning(
            "the largest scout prompt is ~%d tokens; with n_predict=%d it may "
            "overflow ctx_size=%d — lower max_chars_chunk or n_predict",
            estimate, n_predict, ctx_size,
        )


ANALYSIS_COMPLETE_FILE = "analysis_complete.json"


def quality_filter_settings(processing_conf: Mapping[str, Any]) -> dict[str, float]:
    """RU: Фильтры качества нарезки (processing.quality_filters) для отбора.

    EN: The cut stage's quality filters (processing.quality_filters), read so
    selection can enforce them. Applied only after selection they silently
    shrank the output below its target and cost an encode per rejected clip.
    """

    section = processing_conf.get("quality_filters")
    if not isinstance(section, Mapping):
        return {}
    out: dict[str, float] = {}
    for key in ("min_score", "min_duration", "max_duration"):
        try:
            if key in section:
                out[key] = float(section[key])
        except (TypeError, ValueError):
            continue
    return out


def split_by_quality_filters(
    records: Sequence[MomentRecord],
    filters: Mapping[str, float],
) -> tuple[list[MomentRecord], list[tuple[MomentRecord, str]]]:
    """Split records into (eligible, [(rejected, reason)])."""

    kept: list[MomentRecord] = []
    rejected: list[tuple[MomentRecord, str]] = []
    min_score = filters.get("min_score", 0.0)
    min_duration = filters.get("min_duration", 0.0)
    max_duration = filters.get("max_duration", 0.0)
    for record in records:
        duration = record.end - record.start
        if min_score > 0 and record.score < min_score:
            rejected.append((record, "below_min_score"))
        elif min_duration > 0 and duration < min_duration:
            rejected.append((record, "shorter_than_min_duration"))
        elif 0 < max_duration < duration:
            rejected.append((record, "longer_than_max_duration"))
        else:
            kept.append(record)
    return kept, rejected


def _warn_on_unreachable_buckets(quotas: Mapping[str, int], filters: Mapping[str, float]) -> None:
    """Say so when a quota bucket can never pass the duration filters."""

    min_duration = filters.get("min_duration", 0.0)
    max_duration = filters.get("max_duration", 0.0)
    for bucket, count in quotas.items():
        if int(count) <= 0:
            continue
        low, high = clip_type_target_bounds(bucket)
        if (min_duration > 0 and high < min_duration) or (0 < max_duration < low):
            LOGGER.warning(
                "clip type %r (%g-%gs) conflicts with quality_filters "
                "(min_duration=%g, max_duration=%g): its %d slot(s) can only be "
                "filled by other types",
                bucket, low, high, min_duration, max_duration, int(count),
            )


async def run_staged_analysis(
    *,
    transcript_path: Path,
    outdir: Path,
    provider_name: str,
    url: str,
    api_key: str | None,
    roles: LlamaCppRoleMapping,
    llama_cpp_conf: Mapping[str, Any],
    prompts_conf: Mapping[str, Any],
    processing_conf: Mapping[str, Any],
    diarization_path: Path | None = None,
    quiet: bool = False,
    verbose: bool = False,
    progress: bool = False,
) -> list[MomentRecord]:
    """Run the full multi-stage analysis pipeline and write artifacts.

    LLM discovers -> Python proves -> deterministic selector chooses -> LLM
    writes metadata:

    1. scout: candidates with verbatim quotes, ids assigned in Python;
    2. quote verification: exact/fuzzy lookup, hard reject below min_ratio;
    3. cleanup: keep/drop/merge decisions by id (quotes and bounds immutable);
    4. judge: reviews by id in stratified batches (score, title, hook, why);
    5. final gate: min_final_ratio, snap, clamp, quote containment;
    6. deterministic MMR selection under quotas and the overlap policy.

    Besides moments.json and reels.md it writes rejected_candidates.json
    (everything a gate threw out, with the reason) and analysis_metrics.json.
    """

    if not transcript_path.exists():
        raise SystemExit(f"Transcript not found: {transcript_path}")

    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read transcript: {exc}") from exc

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        raise SystemExit("Transcript JSON is missing a segments list")

    duration = data.get("duration", 0.0)
    if not duration and segments:
        try:
            duration = float(segments[-1]["end"])
        except (KeyError, TypeError, ValueError):
            duration = 0.0

    diar = _load_diarization(diarization_path)
    if diar:
        _assign_speakers(segments, diar, prefix=False)

    # Word- and sentence-level timings, used to ground quotes and to anchor
    # clip boundaries to real speech.
    index = TranscriptIndex.from_transcript(data)

    prompt_lang = _normalize_prompt_lang(
        str(prompts_conf.get("language", "auto")),
        str(data.get("language") or ""),
    )
    variant = str(prompts_conf.get("variant", "default")).strip().lower() or "default"
    try:
        clips_per_hour = float(processing_conf.get("clips_per_hour", 0) or 0)
    except (TypeError, ValueError):
        clips_per_hour = 0.0
    quotas = scale_quotas_to_duration(
        _requested_quotas(processing_conf),
        duration_s=float(duration),
        clips_per_hour=clips_per_hour,
    )
    requirements = _build_requirements_text(processing_conf, quotas=quotas)
    scout_requirements = _build_requirements_text(
        processing_conf, quotas=quotas, include_counts=False,
    )
    target_total = sum(max(0, int(v)) for v in quotas.values())
    if target_total <= 0:
        target_total = int(processing_conf.get("reels_count", 4))
    if clips_per_hour > 0:
        LOGGER.info(
            "[analyze] duration=%.0fs -> target clips=%d (%s/hour, quotas=%s)",
            float(duration), target_total, clips_per_hour, quotas,
        )

    analysis_conf = analysis_conf_section(processing_conf)
    json_retries = _conf_int(analysis_conf, "json_retry", 1)
    budget = RetryBudget(_conf_int(analysis_conf, "json_retry_budget", 10))
    cleanup_cap = _conf_int(analysis_conf, "cleanup_cap", _CLEANUP_CAP)
    # Constraining the sampler to each stage's schema is what keeps the JSON
    # parseable; builds that reject a schema step down to a simplified one,
    # and strict_json_schema=false is the escape hatch for the rest.
    strict_schema = _conf_bool(analysis_conf, "strict_json_schema", True)

    def _schemas(
        full: Mapping[str, Any], fallback_key: str,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
        if not strict_schema:
            return None, None
        return full, FALLBACK_SCHEMAS[fallback_key]

    validation_conf = analysis_conf_section(processing_conf, "validation")
    chunk_tolerance_s = _conf_float(validation_conf, "chunk_tolerance_s", 3.0)
    require_overlap = _conf_bool(validation_conf, "require_candidate_overlap", True)
    quote_conf = quote_verification_settings(
        analysis_conf_section(processing_conf, "quote_verification"),
    )
    audio_conf = analysis_conf_section(processing_conf, "audio_features")
    audio_enabled = _conf_bool(audio_conf, "enabled", True)
    context_conf = analysis_conf_section(processing_conf, "episode_context")
    context_enabled = _conf_bool(context_conf, "enabled", True)
    context_max_digest = _conf_int(context_conf, "max_digest_chars", 4000)
    judge_ctx_conf = analysis_conf_section(processing_conf, "judge_context")
    judge_ctx_enabled = _conf_bool(judge_ctx_conf, "enabled", True)
    judge_max_candidates = _conf_int(judge_ctx_conf, "max_candidates", 14)
    judge_head_s = _conf_float(judge_ctx_conf, "head_seconds", 15.0)
    judge_tail_s = _conf_float(judge_ctx_conf, "tail_seconds", 5.0)
    judge_excerpt_chars = _conf_int(judge_ctx_conf, "max_excerpt_chars", 260)
    snap_conf = analysis_conf_section(processing_conf, "boundary_snap")
    snap_enabled = _conf_bool(snap_conf, "enabled", True)
    snap_max_shift_s = _conf_float(snap_conf, "max_shift_s", 3.0)
    scoring_weights = analysis_conf_section(processing_conf, "scoring").get("weights")
    if not isinstance(scoring_weights, Mapping):
        scoring_weights = None
    diversity_conf = analysis_conf_section(processing_conf, "diversity")
    diversity_enabled = _conf_bool(diversity_conf, "enabled", True)
    max_topic_similarity = _conf_float(diversity_conf, "max_topic_similarity", 0.5)
    mmr_lambda = _conf_float(diversity_conf, "mmr_lambda", DEFAULT_MMR_LAMBDA)
    selection_conf = analysis_conf_section(processing_conf, "selection")
    max_overlap_ratio = _conf_float(selection_conf, "max_overlap_ratio", DEFAULT_MAX_OVERLAP_RATIO)

    base_timeout = int(llama_cpp_conf.get("timeout", 900))
    scout_conf = _stage_config(llama_cpp_conf, role="scout", model=roles.scout)
    cleanup_refine_conf = _stage_config(llama_cpp_conf, role="cleanup_refine", model=roles.cleanup_refine)
    judge_metadata_conf = _stage_config(llama_cpp_conf, role="judge_metadata", model=roles.judge_metadata)
    base_url = str(llama_cpp_conf.get("url", url))

    scout_schema, scout_fallback = _schemas(SCOUT_JSON_SCHEMA, "scout")
    cleanup_schema, cleanup_fallback = _schemas(CLEANUP_DECISIONS_SCHEMA, "cleanup")
    judge_schema, judge_fallback = _schemas(JUDGE_REVIEWS_SCHEMA, "judge")
    context_schema, context_fallback = _schemas(EPISODE_CONTEXT_SCHEMA, "context")

    scout_provider = _make_stage_provider(
        provider_name,
        model=roles.scout,
        base_url=base_url,
        stage_conf=scout_conf,
        api_key=api_key,
        json_schema=scout_schema,
        fallback_schema=scout_fallback,
    )
    cleanup_refine_provider = _make_stage_provider(
        provider_name,
        model=roles.cleanup_refine,
        base_url=base_url,
        stage_conf=cleanup_refine_conf,
        api_key=api_key,
        json_schema=cleanup_schema,
        fallback_schema=cleanup_fallback,
    )
    judge_metadata_provider = _make_stage_provider(
        provider_name,
        model=roles.judge_metadata,
        base_url=base_url,
        stage_conf=judge_metadata_conf,
        api_key=api_key,
        json_schema=judge_schema,
        fallback_schema=judge_fallback,
    )
    llm_cache_enabled = _conf_bool(analysis_conf, "llm_cache", True)
    llm_cache_dir = outdir / "llm_cache"
    llm_cache_used: set[str] = set()

    def _cached(provider: LLMProvider, model: str, schema: Mapping[str, Any] | None) -> LLMProvider:
        if not llm_cache_enabled:
            return provider
        namespace = json.dumps([provider_name, model, schema], sort_keys=True, default=str)
        return CachingProvider(provider, llm_cache_dir, namespace=namespace, used=llm_cache_used)

    scout_provider = _cached(scout_provider, roles.scout, scout_schema)
    cleanup_refine_provider = _cached(cleanup_refine_provider, roles.cleanup_refine, cleanup_schema)
    judge_metadata_provider = _cached(judge_metadata_provider, roles.judge_metadata, judge_schema)
    providers: list[LLMProvider] = [scout_provider, cleanup_refine_provider, judge_metadata_provider]

    scout_prompt = _ensure_prompt_text("scout", prompt_lang, variant)
    cleanup_refine_prompt = _ensure_prompt_text("cleanup_refine", prompt_lang, variant)
    judge_metadata_prompt = _ensure_prompt_text("judge_metadata", prompt_lang, variant)

    scout_chunk_seconds = _stage_chunk_seconds(scout_conf, int(llama_cpp_conf.get("chunk_seconds", 900)))
    scout_max_chars = _stage_max_chars(scout_conf, int(llama_cpp_conf.get("max_chars_chunk", 12000)))
    overlap_s = _conf_int(
        analysis_conf, "chunk_overlap_s", adaptive_overlap_seconds(scout_chunk_seconds),
    )
    chunks = build_analysis_chunks(
        segments,
        chunk_seconds=scout_chunk_seconds,
        max_chars=scout_max_chars,
        overlap_seconds=overlap_s,
    )
    _warn_on_context_budget(
        chunks,
        prompt=scout_prompt,
        llama_cpp_conf=llama_cpp_conf,
        n_predict=_conf_int(scout_conf, "n_predict", 4096),
    )
    manifest = {
        "transcript": str(transcript_path.resolve()),
        "duration": float(duration),
        "roles": roles.as_dict(),
        "quotas": quotas,
        "prompt_lang": prompt_lang,
        "prompt_variant": variant,
        "chunk_count": len(chunks),
        "chunk_overlap_s": overlap_s,
        "timing_version": data.get("timing_version", 1),
        "source_audio": data.get("source_audio") or data.get("audio"),
        "language": data.get("language"),
        "language_confidence": data.get("language_confidence"),
        "speaker_aware": bool(diar),
        "quote_verification": quote_conf,
    }
    atomic_write_json(outdir / "analysis_manifest.json", manifest)
    # A marker left by an earlier run must not vouch for this one if it dies.
    (outdir / ANALYSIS_COMPLETE_FILE).unlink(missing_ok=True)
    filters = quality_filter_settings(processing_conf)
    _warn_on_unreachable_buckets(quotas, filters)

    _status(
        f"[analyze] scout={roles.scout} cleanup_refine={roles.cleanup_refine} "
        f"judge_metadata={roles.judge_metadata}",
        quiet=quiet,
    )
    _status(f"[analyze] chunks={len(chunks)}", quiet=quiet)

    scout_temp = _stage_temperature(scout_conf, _default_stage_temperature("scout"))
    cleanup_refine_temp = _stage_temperature(cleanup_refine_conf, _default_stage_temperature("cleanup_refine"))
    judge_metadata_temp = _stage_temperature(judge_metadata_conf, _default_stage_temperature("judge_metadata"))

    scout_timeout = _stage_timeout(scout_conf, base_timeout)
    cleanup_refine_timeout = _stage_timeout(cleanup_refine_conf, base_timeout)
    judge_metadata_timeout = _stage_timeout(judge_metadata_conf, base_timeout)
    default_parallelism = int(llama_cpp_conf.get("scout_parallelism", 1))
    scout_parallelism = _stage_parallelism(scout_conf, default_parallelism)
    stage_parallelism = int(llama_cpp_conf.get("stage_parallelism", default_parallelism))
    cleanup_parallelism = _stage_parallelism(cleanup_refine_conf, stage_parallelism)
    judge_parallelism = _stage_parallelism(judge_metadata_conf, stage_parallelism)
    if provider_name != "llama_cpp":
        scout_parallelism = cleanup_parallelism = judge_parallelism = 1

    stats = {name: StageStats() for name in ("context", "scout", "cleanup", "judge")}
    counts: dict[str, int] = {}
    rejected: list[dict[str, Any]] = []
    started = time.monotonic()

    try:
        episode_context = ""
        if context_enabled:
            # Each stage's grammar only admits its own shape, so the overview
            # needs its own provider with its own schema.
            context_provider = _cached(
                _make_stage_provider(
                    provider_name,
                    model=roles.scout,
                    base_url=base_url,
                    stage_conf=scout_conf,
                    api_key=api_key,
                    json_schema=context_schema,
                    fallback_schema=context_fallback,
                ),
                roles.scout,
                context_schema,
            )
            providers.append(context_provider)
            episode_context = await build_episode_context(
                context_provider,
                index,
                outdir=outdir,
                lang=prompt_lang,
                variant=variant,
                temperature=scout_temp,
                timeout=scout_timeout,
                max_digest_chars=context_max_digest,
                json_retries=json_retries,
                model=roles.scout,
                budget=budget,
                stats=stats["context"],
            )
            if episode_context:
                _status("[analyze] episode context ready", quiet=quiet)

        # -- A/B: discovery -------------------------------------------------
        scouted_candidates = await scout_candidates(
            scout_provider,
            chunks,
            requirements=scout_requirements,
            prompt=scout_prompt,
            temperature=scout_temp,
            timeout=scout_timeout,
            progress=bool(progress and verbose and not quiet),
            parallelism=scout_parallelism,
            json_retries=json_retries,
            chunk_tolerance_s=chunk_tolerance_s,
            episode_context=episode_context,
            budget=budget,
            stats=stats["scout"],
        )
        atomic_write_json(outdir / "scout_candidates.json", build_candidate_json(scouted_candidates))
        counts["scouted"] = len(scouted_candidates)

        # -- C: Python proves -------------------------------------------------
        # Verify quotes before any further LLM call: a candidate whose quote
        # is not in the transcript is not worth cleanup or judge tokens.
        # Refinement widens bounds over the located quote.
        verified = apply_quote_verification(scouted_candidates, index, **quote_conf)
        measured = [r for r in verified if r.quote_match_ratio is not None]
        quote_stats = {
            "measured": len(measured),
            "exact": sum(1 for r in measured if r.quote_match_method == METHOD_EXACT),
            "low_confidence": sum(
                1 for r in measured
                if (r.quote_match_ratio or 0.0) < quote_conf["min_final_ratio"]
            ),
        }
        if quote_conf["enabled"]:
            verified, unproven = split_by_quote_ratio(verified, quote_conf["min_ratio"])
            rejected += _rejection_rows(unproven, stage="scout", reason="quote_not_in_transcript")
            if unproven:
                LOGGER.info(
                    "quote gate: rejected %d of %d scouted candidate(s) whose quote "
                    "was not found in the transcript (min_ratio=%.2f)",
                    len(unproven), len(scouted_candidates), quote_conf["min_ratio"],
                )
        counts["after_quote_gate"] = len(verified)

        # Limit total candidates before cleanup to stay within ctx=8192.
        # Dedupe first: chunks overlap by design, so the same strong moment is
        # scouted twice and would otherwise burn two of the capped slots.
        cleanup_total_cap = max(cleanup_cap, target_total * 2)
        cleanup_input = dedupe_moments(verified)
        counts["after_dedupe"] = len(cleanup_input)
        if len(cleanup_input) > cleanup_total_cap:
            cleanup_input = sorted(cleanup_input, key=ranking_value, reverse=True)[:cleanup_total_cap]
        if len(cleanup_input) < len(verified):
            LOGGER.info(
                "pre-cleanup: %d verified -> %d candidates (deduped, capped at %d)",
                len(verified), len(cleanup_input), cleanup_total_cap,
            )
        counts["cleanup_input"] = len(cleanup_input)

        cleaned_candidates = await cleanup_and_refine_candidates(
            cleanup_refine_provider,
            cleanup_input,
            requirements=requirements,
            prompt=cleanup_refine_prompt,
            temperature=cleanup_refine_temp,
            timeout=cleanup_refine_timeout,
            max_items=max(12, target_total * 3),
            json_retries=json_retries,
            batch_size=cleanup_cap,
            parallelism=cleanup_parallelism,
            budget=budget,
            stats=stats["cleanup"],
        )
        cleaned_candidates = _guard_stage_output(
            cleaned_candidates, cleanup_input, stage="cleanup", enabled=require_overlap,
        )
        counts["after_cleanup"] = len(cleaned_candidates)
        # Cleanup is meant to drop weak candidates, but a misbehaving model
        # can also lose good ones. If the pool fell below what the target
        # needs (with slack for overlap/diversity losses at selection),
        # restore the best dropped candidates that don't overlap anything the
        # cleanup kept.
        pool_floor = max(target_total + max(3, target_total // 3), len(cleaned_candidates))
        if len(cleaned_candidates) < pool_floor:
            kept = list(cleaned_candidates)
            for candidate in sorted(cleanup_input, key=ranking_value, reverse=True):
                if len(kept) >= pool_floor:
                    break
                if not any(overlap_ratio_of_shorter(candidate, existing) > 0 for existing in kept):
                    kept.append(candidate)
            if len(kept) > len(cleaned_candidates):
                LOGGER.info(
                    "cleanup pool top-up: %d -> %d candidates (target %d)",
                    len(cleaned_candidates), len(kept), target_total,
                )
                cleaned_candidates = kept
        # Measure the audio once the list is short: the judge and the final
        # ranking both get to use it.
        if audio_enabled:
            source_audio = resolve_source_audio(data)
            if source_audio is None:
                LOGGER.info("no source audio next to the transcript; skipping audio features")
            else:
                cleaned_candidates = annotate_records_with_audio(
                    cleaned_candidates,
                    source_audio,
                    noise_db=_conf_float(audio_conf, "silence_noise_db", -30.0),
                    silence_min_s=_conf_float(audio_conf, "silence_min_s", 0.35),
                    timeout_s=_conf_int(audio_conf, "timeout_s", 30),
                    max_candidates=_conf_int(audio_conf, "max_candidates", 40),
                    max_workers=_conf_int(audio_conf, "parallelism", 4),
                    cache_path=(
                        outdir / "audio_features_cache.json"
                        if _conf_bool(audio_conf, "cache", True)
                        else None
                    ),
                )
        atomic_write_json(outdir / "cleaned_candidates.json", build_candidate_json(cleaned_candidates))

        # -- D: judge in stratified batches --------------------------------
        # One prompt only fits ~14 candidates inside ctx_size=8192. Each
        # batch gets the same spread of quality, so the absolute scores stay
        # comparable; the global comparison is the deterministic ranking.
        judge_batches = stratified_batches(cleaned_candidates, max(1, judge_max_candidates))
        counts["judge_input"] = len(cleaned_candidates)

        async def _judge_batch(batch: Sequence[MomentRecord]) -> list[MomentRecord]:
            try:
                return await judge_candidates(
                    judge_metadata_provider,
                    batch,
                    requirements=requirements,
                    prompt=judge_metadata_prompt,
                    temperature=judge_metadata_temp,
                    timeout=judge_metadata_timeout,
                    json_retries=json_retries,
                    episode_context=episode_context,
                    candidates_payload=(
                        build_judge_payload(
                            batch,
                            index,
                            max_candidates=len(batch),
                            head_seconds=judge_head_s,
                            tail_seconds=judge_tail_s,
                            max_excerpt_chars=judge_excerpt_chars,
                        )
                        if judge_ctx_enabled
                        else None
                    ),
                    budget=budget,
                    stats=stats["judge"],
                )
            except Exception as exc:
                LOGGER.warning("judge batch failed; keeping its candidates unjudged: %s", exc)
                return list(batch)

        judged_batches = await _gather_batches(
            judge_batches, _judge_batch, parallelism=judge_parallelism,
        )
        judged_candidates = [record for batch in judged_batches for record in batch]
        if cleaned_candidates and not judged_candidates:
            LOGGER.warning(
                "the judge rejected every candidate; falling back to the pre-judge pool",
            )
            judged_candidates = list(cleaned_candidates)
        judged_candidates = _guard_stage_output(
            judged_candidates, cleaned_candidates, stage="judge", enabled=require_overlap,
        )
        counts["after_judge"] = len(judged_candidates)

        # -- E: final gate --------------------------------------------------
        # Quotes cannot change after the scout, so no re-verification: only
        # the stricter final threshold, then boundary work that must keep the
        # located quote inside the clip.
        if quote_conf["enabled"]:
            judged_candidates, weak = split_by_quote_ratio(
                judged_candidates, quote_conf["min_final_ratio"],
            )
            rejected += _rejection_rows(weak, stage="final", reason="quote_low_confidence")
        counts["after_final_gate"] = len(judged_candidates)

        before_snap = {id(record): (record.start, record.end) for record in judged_candidates}
        snapped = snap_records(
            judged_candidates, index, enabled=snap_enabled, max_shift_s=snap_max_shift_s,
        )
        shifts: list[float] = []
        for original, moved in zip(judged_candidates, snapped):
            start0, end0 = before_snap[id(original)]
            shifts += [abs(moved.start - start0), abs(moved.end - end0)]
        clamped = clamp_records_to_episode(snapped, float(duration))
        contained, lost = enforce_quote_containment(clamped, duration=float(duration))
        rejected += _rejection_rows(lost, stage="final", reason="quote_outside_clip")
        annotated = annotate_speech_rate(contained, index)
        typed = assign_clip_types(annotated, quotas)
        # The cut stage's own filters, enforced here so every selected slot
        # goes to a clip that will actually be cut.
        typed, filtered_out = split_by_quality_filters(typed, filters)
        for record, reason in filtered_out:
            rejected += _rejection_rows([record], stage="selection", reason=reason)
        counts["after_quality_filters"] = len(typed)

        # -- F: deterministic selection, exactly once ------------------------
        # Ranking twice used to feed the combined priority back in as the next
        # pass's base score.
        selected = rank_moments(
            typed,
            clip_type_quotas=quotas,
            scoring_weights=scoring_weights,
            diversity_enabled=diversity_enabled,
            max_topic_similarity=max_topic_similarity,
            # With duration-scaled quotas the total is the promise and the mix
            # is a preference; fixed legacy quotas stay strict.
            fill_to_total=target_total if clips_per_hour > 0 else None,
            max_overlap_ratio=max_overlap_ratio,
            mmr_lambda=mmr_lambda,
        )
        if typed and not selected:
            # Every candidate fell outside the configured quotas. Emitting
            # them unranked would quietly override the config, so report it.
            LOGGER.warning(
                "no candidate matched the configured clip quotas %s "
                "(%d candidates, types: %s)",
                quotas,
                len(typed),
                sorted({record.clip_type for record in typed}),
            )
        final_moments = finalize_moment_list(selected)
    finally:
        for provider in providers:
            await close_provider(provider)

    final_payload = [moment.to_dict() for moment in final_moments]
    atomic_write_json(outdir / "moments.json", final_payload)
    (outdir / "reels.md").write_text(
        render_reels_summary_markdown(final_moments),
        encoding="utf-8",
    )
    atomic_write_json(outdir / "rejected_candidates.json", rejected)

    counts["final"] = len(final_moments)
    rejection_reasons: dict[str, int] = {}
    for row in rejected:
        reason = str(row.get("rejection_reason", ""))
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    hours = float(duration) / 3600.0 if duration else 0.0
    metrics = {
        "duration_s": float(duration),
        "target_clips": target_total,
        "elapsed_s": round(time.monotonic() - started, 2),
        "counts": counts,
        "stages": {name: stage.as_dict() for name, stage in stats.items()},
        "scout_candidates_per_hour": _rate(counts.get("scouted", 0), hours),
        "candidate_survival_rate": {
            "quote_gate": _rate(counts.get("after_quote_gate", 0), counts.get("scouted", 0)),
            "cleanup": _rate(counts.get("after_cleanup", 0), counts.get("cleanup_input", 0)),
            "judge": _rate(counts.get("after_judge", 0), counts.get("judge_input", 0)),
        },
        "quote_exact_match_rate": _rate(quote_stats["exact"], quote_stats["measured"]),
        "quote_low_confidence_rate": _rate(quote_stats["low_confidence"], quote_stats["measured"]),
        "boundary_shift_seconds": {
            "mean": round(sum(shifts) / len(shifts), 3) if shifts else 0.0,
            "p95": _percentile(shifts, 0.95),
        },
        "duplicate_rate": _rate(
            counts.get("after_quote_gate", 0) - counts.get("after_dedupe", 0),
            counts.get("after_quote_gate", 0),
        ),
        "topic_diversity": _topic_diversity(final_moments),
        "quota_fill_rate": _rate(len(final_moments), target_total),
        "json_retries": {"used": budget.used, "refused": budget.refused, "budget": budget.total},
        "llm_cache_hits": sum(getattr(p, "hits", 0) for p in providers),
        "rejections": rejection_reasons,
    }
    atomic_write_json(outdir / "analysis_metrics.json", metrics)
    if llm_cache_enabled:
        # Answers this run did not ask for belong to prompts that no longer
        # exist; keeping them would only grow the folder forever.
        CachingProvider.prune(llm_cache_dir, llm_cache_used)
    # Written last: its presence means the analysis ran to the end, so an
    # empty moments.json is a real result ("nothing worth cutting") rather
    # than the placeholder a crash leaves behind.
    atomic_write_json(
        outdir / ANALYSIS_COMPLETE_FILE,
        {"status": "ok", "moments": len(final_moments), "finished_at": time.time()},
    )

    if not quiet:
        LOGGER.info("[analyze] moments=%d", len(final_moments))
        LOGGER.info("[analyze] saved=%s", outdir / "moments.json")

    return final_moments


async def find_moments(
    provider: LLMProvider,
    segments: list[dict[str, Any]],
    duration: float,
    r_min: int,
    r_max: int,
    count: int,
    chunk_sec: int,
    max_ch: int,
    timeout: int,
    progress: bool = False,
    *,
    ch_prompt: str,
    select_prompt: str,
    stories_count: int = 0,
    reels_count: int = 0,
    long_reels_count: int = 0,
    highlights_moments: int = 0,
) -> list[Moment]:
    """Backward-compatible single-stage helper used by older tests."""

    chunks = chunk_segments_by_time(segments, max(1, int(chunk_sec)))
    candidates: list[MomentRecord] = []
    it = enumerate(chunks, 1)
    if progress:
        it = tqdm(it, total=len(chunks), desc="analyze")

    reqs = []
    if stories_count > 0:
        reqs.append(f"Stories (up to 15s): {stories_count}")
    if reels_count > 0:
        reqs.append(f"Reels (up to 60s): {reels_count}")
    if long_reels_count > 0:
        reqs.append(f"Long Reels (up to 180s): {long_reels_count}")
    if highlights_moments > 0:
        reqs.append(f"Hot moments for highlights: {highlights_moments}")
    reqs_str = "\n".join(reqs) if reqs else f"Viral moments ({r_min}-{r_max}s): {count}"

    for idx, ch in it:
        ch_txt = segments_to_compact_text(ch, max_ch)
        prompt = _render_prompt(
            ch_prompt,
            {
                "r_min": str(r_min),
                "r_max": str(r_max),
                "transcript": ch_txt,
                "requirements": reqs_str,
                "chunk_json": json.dumps({"chunk_id": idx, "text": ch_txt}, ensure_ascii=False),
            },
        )
        resp = await get_llm_json(provider, prompt, 0.3, timeout)
        chunk_moments = _parse_candidate_response(resp, stage="scout")
        for moment in chunk_moments:
            candidates.append(moment)

    if not candidates:
        return []

    # Use the old select prompt only as a compatibility anchor for prompt tests.
    _ = select_prompt
    quotas = {"reel": count}
    ranked = rank_moments(candidates, clip_type_quotas=quotas)
    # Legacy path: no transcript index to prove quotes against, so a missing
    # quote is tolerated here — but, as everywhere, never made up.
    final = finalize_moment_list(ranked, require_quote=False)
    return final[: max(0, int(count))]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the analysis stage."""

    ap = argparse.ArgumentParser(
        description="Analyze transcript with local staged llama.cpp models to find viral moments.",
    )
    ap.add_argument("--transcript", type=Path, required=True, help="Path to transcript JSON file")
    ap.add_argument("--outdir", type=Path, default=Path("out"), help="Output directory")
    ap.add_argument(
        "--provider",
        choices=("llama_cpp", "openai", "anthropic", "gemini"),
        default="llama_cpp",
        help="LLM provider to use (cloud providers are legacy compatibility only)",
    )
    ap.add_argument("--api-key", help="Optional override for legacy cloud providers")
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:8080/v1/chat/completions",
        help="llama.cpp API URL",
    )
    ap.add_argument("--model", help="Legacy single-model mode (maps to all roles)")
    ap.add_argument("--scout-model", help="Scout role model")
    ap.add_argument("--cleanup-model", help="Cleanup/refine role model")
    ap.add_argument("--judge-model", help="Judge/metadata role model")
    ap.add_argument("--temperature", type=float, default=0.25, help="Base temperature")
    ap.add_argument("--reels", type=int, default=4, help="Number of reels to generate")
    ap.add_argument("--stories-count", type=int, default=0, help="Number of stories (up to 15s)")
    ap.add_argument("--reels-count", type=int, default=0, help="Number of reels (up to 60s)")
    ap.add_argument("--long-reels-count", type=int, default=0, help="Number of long reels (up to 180s)")
    ap.add_argument("--highlights-moments", type=int, default=0, help="Number of hot moments for highlights")
    ap.add_argument("--reel-min", type=int, default=30, help="Minimum reel duration (seconds)")
    ap.add_argument("--reel-max", type=int, default=60, help="Maximum reel duration (seconds)")
    ap.add_argument("--chunk-seconds", type=int, default=900, help="Chunk size for scouting")
    ap.add_argument("--max_chars_chunk", type=int, default=12000, help="Max chars per chunk")
    ap.add_argument("--timeout", type=int, default=900, help="LLM request timeout")
    ap.add_argument("--prompt-lang", default="auto", help="Prompt language: ru|en|auto")
    ap.add_argument("--prompt-variant", default="default", help="Prompt variant: default|a|b")
    ap.add_argument("--diarization", type=Path, help="Optional diarization.json for speaker tags")
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    ap.add_argument("--llama-watchdog", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable llama.cpp stall watchdog")
    ap.add_argument("--llama-first-token-timeout", type=int, default=120, help="No output timeout before first token")
    ap.add_argument("--llama-stall-timeout", type=int, default=120, help="No output timeout while streaming")
    ap.add_argument("--llama-log-interval", type=int, default=10, help="Progress heartbeat interval")
    ap.add_argument("--llama-max-retries", type=int, default=2, help="Retries on stall/timeout")
    ap.add_argument(
        "--llama-fallback-model",
        action="append",
        default=[],
        help="Fallback model to try on stall/timeout (can be repeated)",
    )
    return ap.parse_args(argv)


def _resolve_role_mapping(args: argparse.Namespace, conf_model: str | None = None) -> LlamaCppRoleMapping:
    if args.model:
        legacy_model = str(args.model).strip()
        role_map = {
            "scout": args.scout_model or legacy_model,
            "cleanup_refine": args.cleanup_model or legacy_model,
            "judge_metadata": args.judge_model or legacy_model,
        }
        return resolve_llama_cpp_role_mapping({"llama_cpp": {"roles": role_map}})
    role_map = {
        "scout": args.scout_model or conf_model or "gemma4",
        "cleanup_refine": args.cleanup_model or conf_model or "gemma4",
        "judge_metadata": args.judge_model or conf_model or "gemma4",
    }
    return resolve_llama_cpp_role_mapping({"llama_cpp": {"roles": role_map}})


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the analysis stage."""

    args = parse_args(argv)

    global LOGGER
    LOGGER = setup_logging(verbose=bool(args.verbose), quiet=bool(args.quiet))

    if not args.transcript.exists():
        raise SystemExit(f"Transcript not found: {args.transcript}")

    try:
        data = json.loads(args.transcript.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read transcript: {exc}") from exc

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    prompts_conf = {
        "language": args.prompt_lang,
        "variant": args.prompt_variant,
    }
    processing_conf = {
        "reels_count": args.reels,
        "reel_min_duration": args.reel_min,
        "reel_max_duration": args.reel_max,
        "clips": {
            "stories": {"count": args.stories_count, "max_duration": 15},
            "reels": {"count": args.reels_count, "max_duration": 60},
            "long_reels": {"count": args.long_reels_count, "max_duration": 180},
            "highlights": {"moments_count": args.highlights_moments},
        },
    }
    llama_cpp_conf = {
        "url": args.url,
        "timeout": args.timeout,
        "temperature": args.temperature,
        "chunk_seconds": args.chunk_seconds,
        "max_chars_chunk": args.max_chars_chunk,
        "watchdog": {
            "enabled": bool(args.llama_watchdog),
            "first_token_timeout": args.llama_first_token_timeout,
            "stall_timeout": args.llama_stall_timeout,
            "log_interval": args.llama_log_interval,
            "max_retries": args.llama_max_retries,
        },
        "fallback_models": list(args.llama_fallback_model or []),
    }

    transcript_lang = data.get("language")
    prompt_lang = _normalize_prompt_lang(args.prompt_lang, transcript_lang if isinstance(transcript_lang, str) else None)
    variant = str(args.prompt_variant).strip().lower() or "default"

    roles = _resolve_role_mapping(args, conf_model=args.model)

    proc: subprocess.Popen | None = None
    try:
        managed_by_pipeline = os.environ.get(ENV_MANAGED_BY_PIPELINE) == "1"
        local = parse_local_llama_cpp_host_port(args.url) if args.url else None
        if args.provider == "llama_cpp" and local and not managed_by_pipeline:
            host, port = local
            proc = llama_cpp_start(host=host, port=port, service_conf={})

        final_moments = asyncio.run(run_staged_analysis(
            transcript_path=args.transcript,
            outdir=outdir,
            provider_name=args.provider,
            url=args.url,
            api_key=args.api_key,
            roles=roles,
            llama_cpp_conf=llama_cpp_conf,
            prompts_conf={
                **prompts_conf,
                "language": prompt_lang,
                "variant": variant,
            },
            processing_conf=processing_conf,
            diarization_path=args.diarization,
            quiet=bool(args.quiet),
            verbose=bool(args.verbose),
            progress=bool(args.verbose and not args.quiet),
        ))

        _status(f"[analyze] moments={len(final_moments)}", quiet=bool(args.quiet))
    finally:
        if proc:
            llama_cpp_stop(proc)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        if LOGGER:
            LOGGER.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        if LOGGER:
            LOGGER.error("Analysis failed: %s", exc)
        else:
            print(f"Analysis failed: {exc}", file=sys.stderr)
        if os.environ.get("DEBUG_FORGE") == "1":
            import traceback

            traceback.print_exc()
        sys.exit(1)
