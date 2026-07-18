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
import json
import os
import subprocess
import sys
from dataclasses import asdict
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
from podcast_reels_forge.analysis.chunking import build_analysis_chunks
from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.analysis.metadata import finalize_moment_list
from podcast_reels_forge.analysis.ranking import (
    dedupe_moments,
    rank_moments,
    ranking_value,
)
from podcast_reels_forge.analysis.serializers import atomic_write_json
from podcast_reels_forge.analysis.transcript_index import TranscriptIndex
from podcast_reels_forge.analysis.validation import (
    annotate_speech_rate,
    apply_quote_verification,
    clamp_record_to_window,
    clamp_records_to_episode,
    filter_nonoverlapping_outputs,
    quote_verification_settings,
    snap_records,
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
)
from podcast_reels_forge.llm.schemas import MOMENTS_JSON_SCHEMA
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

# Upper bound on candidates handed to the cleanup stage, so the rendered
# prompt stays inside ctx_size=8192.
_CLEANUP_CAP = 25

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


async def get_llm_json(
    provider: LLMProvider,
    prompt: str,
    temperature: float,
    timeout: int,
    *,
    retries: int = 0,
) -> dict[str, Any] | list[Any]:
    """Get JSON from an LLM response, logging a safe preview on failure.

    An unparseable answer costs a whole chunk, so the request is re-issued up
    to ``retries`` times with an explicit note about the malformed output.
    """

    attempts = max(1, 1 + int(retries))
    for attempt in range(1, attempts + 1):
        prompt_text = prompt if attempt == 1 else prompt + _JSON_RETRY_NOTE
        raw = await provider.generate(prompt_text, temperature=temperature, timeout=timeout)
        try:
            return extract_first_json_value(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            preview = raw[:500].replace("\n", "\\n") if isinstance(raw, str) else str(raw)
            if attempt < attempts:
                LOGGER.warning(
                    "Failed to parse JSON from LLM output (attempt %d/%d), retrying "
                    "(raw preview: %s)",
                    attempt,
                    attempts,
                    preview,
                )
                continue
            LOGGER.warning(
                "Failed to parse JSON from LLM output after %d attempt(s); "
                "returning [] (raw preview: %s)",
                attempts,
                preview,
            )
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


def _build_requirements_text(processing_conf: Mapping[str, Any]) -> str:
    clips_conf = processing_conf.get("clips", {})
    if not isinstance(clips_conf, Mapping):
        clips_conf = {}

    parts: list[str] = []
    if isinstance(clips_conf.get("stories"), Mapping):
        stories = clips_conf["stories"]
        parts.append(
            f"Stories: {stories.get('count', 0)} clips up to {stories.get('max_duration', 15)}s",
        )
    if isinstance(clips_conf.get("reels"), Mapping):
        reels = clips_conf["reels"]
        parts.append(
            f"Reels: {reels.get('count', 0)} clips up to {reels.get('max_duration', 60)}s",
        )
    if isinstance(clips_conf.get("long_reels"), Mapping):
        long_reels = clips_conf["long_reels"]
        parts.append(
            "Long reels: "
            f"{long_reels.get('count', 0)} clips up to {long_reels.get('max_duration', 180)}s",
        )
    if isinstance(clips_conf.get("highlights"), Mapping):
        highlights = clips_conf["highlights"]
        parts.append(
            f"Highlights: {highlights.get('moments_count', 0)} moments",
        )

    if not parts:
        reel_min = processing_conf.get("reel_min_duration", 30)
        reel_max = processing_conf.get("reel_max_duration", 60)
        parts.append(f"Reels: 4 clips of {reel_min}-{reel_max}s")
    return "\n".join(parts)


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
) -> MomentRecord:
    payload = {
        **record.to_dict(),
        "source_chunk_ids": list(dict.fromkeys([*record.source_chunk_ids, chunk_id])),
    }
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
) -> LLMProvider:
    return _provider_for_role(
        provider_name,
        model,
        base_url=base_url,
        role_conf=stage_conf,
        api_key=api_key,
        json_schema=json_schema,
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
) -> list[MomentRecord]:
    """Scout each chunk for candidates.

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
                resp = await get_llm_json(
                    provider, prompt_text, temperature, timeout, retries=json_retries,
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
            dropped = 0
            for candidate in chunk_candidates:
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
                        dropped += 1
                        continue
                    candidate = clamped
                out.append(
                    _attach_chunk_metadata(
                        candidate,
                        chunk_id=chunk_id,
                        speaker_set=getattr(chunk, "speaker_set", ()),
                    ),
                )
            if dropped:
                LOGGER.info(
                    "%s: dropped %d candidate(s) outside the chunk window",
                    chunk_id, dropped,
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
) -> list[MomentRecord]:
    cleaned_input = dedupe_moments(candidates)
    if not cleaned_input:
        return []

    sorted_candidates = sorted(
        cleaned_input,
        key=lambda record: (-ranking_value(record), record.start, record.end),
    )[:max(1, int(max_items))]

    prompt_text = _render_prompt(
        prompt,
        _prompt_payload(
            requirements=requirements,
            candidates=sorted_candidates,
        ),
    )
    resp = await get_llm_json(
        provider, prompt_text, temperature, timeout, retries=json_retries,
    )
    refined = _parse_candidate_response(resp, stage="cleanup_refine")
    if refined:
        return dedupe_moments(refined)
    return list(sorted_candidates)


def _keep_judge_score(record: MomentRecord) -> MomentRecord:
    """Record the judge's own rating alongside the scout's."""

    payload = {**record.to_dict(), "judge_score": float(record.score)}
    return coerce_moment_record(payload) or record


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
) -> list[MomentRecord]:
    """Re-rate candidates globally.

    Returns records as the judge rated them — ranking, quota selection and
    metadata finalization all happen once, in the caller, so the combined
    priority is never computed on top of itself.
    """

    if not candidates:
        return []

    prompt_text = _render_prompt(
        prompt,
        _prompt_payload(
            requirements=requirements,
            candidates=candidates,
            candidates_payload=candidates_payload,
            episode_context=episode_context,
        ),
    )
    resp = await get_llm_json(
        provider, prompt_text, temperature, timeout, retries=json_retries,
    )
    judged = _parse_candidate_response(resp, stage="judge_metadata")
    if not judged:
        return list(candidates)
    return [_keep_judge_score(record) for record in judged]


def build_transcript_digest(index: TranscriptIndex, *, max_chars: int = 4000) -> str:
    """RU: Выжимка эпизода: по одному предложению на окно ~2 минуты.

    EN: An episode digest: roughly one sentence per two-minute window, so the
    whole episode fits into a single context-sized prompt.
    """

    if not index.sentences:
        return ""

    episode_end = index.sentences[-1].end
    window = max(60.0, episode_end / 40.0)
    picked: list[str] = []
    next_slot = index.sentences[0].start
    for sentence in index.sentences:
        if sentence.start < next_slot:
            continue
        picked.append(sentence.text)
        next_slot = sentence.start + window

    digest = " ".join(picked).strip()
    if len(digest) > max_chars:
        digest = digest[:max_chars].rsplit(" ", 1)[0] + "…"
    return digest


def format_episode_context(payload: Mapping[str, Any]) -> str:
    """Render the episode overview as a prompt section."""

    summary = str(payload.get("summary", "")).strip()
    topics = payload.get("topics")
    tone = str(payload.get("tone", "")).strip()
    speakers = payload.get("speakers")

    lines: list[str] = []
    if summary:
        lines.append(summary)
    if isinstance(topics, list) and topics:
        joined = ", ".join(str(topic).strip() for topic in topics if str(topic).strip())
        if joined:
            lines.append(f"Темы эпизода / Episode topics: {joined}")
    if tone:
        lines.append(f"Тональность / Tone: {tone}")
    if isinstance(speakers, list) and speakers:
        joined = ", ".join(str(name).strip() for name in speakers if str(name).strip())
        if joined:
            lines.append(f"Участники / Speakers: {joined}")

    if not lines:
        return ""
    return "# Контекст эпизода / Episode context\n" + "\n".join(lines)


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
) -> str:
    """Summarize the episode once, so the scout can judge moments in context.

    A moment can look striking inside its own chunk and be unremarkable for
    the episode; the scout has no way to tell without this. Entirely
    best-effort: any failure returns an empty string and the prompts render
    without the section.
    """

    cache_path = outdir / "episode_context.json"
    cached = _read_json_if_valid(cache_path)
    if isinstance(cached, dict) and cached.get("summary"):
        return format_episode_context(cached)

    digest = build_transcript_digest(index, max_chars=max_digest_chars)
    if not digest:
        return ""

    try:
        prompt = _load_prompt(lang=lang, variant=variant, name="context")
    except FileNotFoundError:
        LOGGER.info("no episode-context prompt for lang=%s; skipping", lang)
        return ""

    try:
        resp = await get_llm_json(
            provider,
            _render_prompt(prompt, {"transcript_digest": digest}),
            temperature,
            timeout,
            retries=json_retries,
        )
    except Exception as exc:
        LOGGER.warning("episode context failed; continuing without it: %s", exc)
        return ""

    if not isinstance(resp, dict) or not resp.get("summary"):
        LOGGER.info("episode context returned no summary; continuing without it")
        return ""

    atomic_write_json(cache_path, resp)
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
            "start": round(record.start, 3),
            "end": round(record.end, 3),
            "clip_type": record.clip_type,
            "title": record.title,
            "quote": record.quote,
            "why": record.why,
            "score": record.score,
        }
        if record.hook:
            item["hook"] = record.hook
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

    Falls back to the stage input if the guard would empty the list, so a
    misbehaving model costs precision rather than the whole episode.
    """

    if not enabled or not outputs:
        return list(outputs)

    kept = filter_nonoverlapping_outputs(outputs, inputs)
    if len(kept) == len(outputs):
        return kept
    if not kept:
        LOGGER.warning(
            "%s returned %d record(s), none overlapping its input; keeping the input",
            stage, len(outputs),
        )
        return list(inputs)
    LOGGER.warning(
        "%s: dropped %d record(s) that overlapped no input candidate",
        stage, len(outputs) - len(kept),
    )
    return kept


def _ensure_prompt_text(stage: str, lang: str, variant: str) -> str:
    return _load_prompt(lang=lang, variant=variant, name=_stage_name_to_prompt_name(stage))


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
    """Run the full multi-stage analysis pipeline and write artifacts."""

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
    requirements = _build_requirements_text(processing_conf)
    quotas = _requested_quotas(processing_conf)
    target_total = sum(max(0, int(v)) for v in quotas.values())
    if target_total <= 0:
        target_total = int(processing_conf.get("reels_count", 4))

    analysis_conf = analysis_conf_section(processing_conf)
    json_retries = _conf_int(analysis_conf, "json_retry", 1)
    cleanup_cap = _conf_int(analysis_conf, "cleanup_cap", _CLEANUP_CAP)
    # Constraining the sampler to the moments schema is what keeps the JSON
    # parseable; the permissive schema stays available as an escape hatch for
    # llama.cpp builds whose grammar converter chokes on it.
    response_schema = (
        MOMENTS_JSON_SCHEMA
        if _conf_bool(analysis_conf, "strict_json_schema", True)
        else None
    )
    validation_conf = analysis_conf_section(processing_conf, "validation")
    chunk_tolerance_s = _conf_float(validation_conf, "chunk_tolerance_s", 3.0)
    require_overlap = _conf_bool(validation_conf, "require_candidate_overlap", True)
    quote_conf = quote_verification_settings(
        analysis_conf_section(processing_conf, "quote_verification"),
    )
    audio_conf = analysis_conf_section(processing_conf, "audio_features")
    audio_enabled = _conf_bool(audio_conf, "enabled", True)
    audio_noise_db = _conf_float(audio_conf, "silence_noise_db", -30.0)
    audio_silence_min_s = _conf_float(audio_conf, "silence_min_s", 0.35)
    audio_timeout_s = _conf_int(audio_conf, "timeout_s", 30)
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

    base_timeout = int(llama_cpp_conf.get("timeout", 900))
    scout_conf = _stage_config(llama_cpp_conf, role="scout", model=roles.scout)
    cleanup_refine_conf = _stage_config(llama_cpp_conf, role="cleanup_refine", model=roles.cleanup_refine)
    judge_metadata_conf = _stage_config(llama_cpp_conf, role="judge_metadata", model=roles.judge_metadata)

    scout_provider = _make_stage_provider(
        provider_name,
        model=roles.scout,
        base_url=str(llama_cpp_conf.get("url", url)),
        stage_conf=scout_conf,
        api_key=api_key,
        json_schema=response_schema,
    )
    cleanup_refine_provider = _make_stage_provider(
        provider_name,
        model=roles.cleanup_refine,
        base_url=str(llama_cpp_conf.get("url", url)),
        stage_conf=cleanup_refine_conf,
        api_key=api_key,
        json_schema=response_schema,
    )
    judge_metadata_provider = _make_stage_provider(
        provider_name,
        model=roles.judge_metadata,
        base_url=str(llama_cpp_conf.get("url", url)),
        stage_conf=judge_metadata_conf,
        api_key=api_key,
        json_schema=response_schema,
    )

    scout_prompt = _ensure_prompt_text("scout", prompt_lang, variant)
    cleanup_refine_prompt = _ensure_prompt_text("cleanup_refine", prompt_lang, variant)
    judge_metadata_prompt = _ensure_prompt_text("judge_metadata", prompt_lang, variant)

    scout_chunk_seconds = _stage_chunk_seconds(scout_conf, int(llama_cpp_conf.get("chunk_seconds", 900)))
    scout_max_chars = _stage_max_chars(scout_conf, int(llama_cpp_conf.get("max_chars_chunk", 12000)))
    chunks = build_analysis_chunks(
        segments,
        chunk_seconds=scout_chunk_seconds,
        max_chars=scout_max_chars,
        overlap_seconds=max(15, scout_chunk_seconds // 8),
    )
    manifest = {
        "transcript": str(transcript_path.resolve()),
        "duration": float(duration),
        "roles": roles.as_dict(),
        "quotas": quotas,
        "prompt_lang": prompt_lang,
        "prompt_variant": variant,
        "chunk_count": len(chunks),
        "timing_version": data.get("timing_version", 1),
        "source_audio": data.get("source_audio") or data.get("audio"),
        "language": data.get("language"),
        "language_confidence": data.get("language_confidence"),
        "speaker_aware": bool(diar),
    }
    atomic_write_json(outdir / "analysis_manifest.json", manifest)

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
    scout_parallelism = _stage_parallelism(
        scout_conf,
        int(llama_cpp_conf.get("scout_parallelism", 1)),
    )
    if provider_name != "llama_cpp":
        scout_parallelism = 1

    episode_context = ""
    if context_enabled:
        episode_context = await build_episode_context(
            scout_provider,
            index,
            outdir=outdir,
            lang=prompt_lang,
            variant=variant,
            temperature=scout_temp,
            timeout=scout_timeout,
            max_digest_chars=context_max_digest,
            json_retries=json_retries,
        )
        if episode_context:
            _status("[analyze] episode context ready", quiet=quiet)

    scouted_candidates = await scout_candidates(
        scout_provider,
        chunks,
        requirements=requirements,
        prompt=scout_prompt,
        temperature=scout_temp,
        timeout=scout_timeout,
        progress=bool(progress and verbose and not quiet),
        parallelism=scout_parallelism,
        json_retries=json_retries,
        chunk_tolerance_s=chunk_tolerance_s,
        episode_context=episode_context,
    )
    atomic_write_json(outdir / "scout_candidates.json", build_candidate_json(scouted_candidates))

    # Limit total candidates before cleanup to stay within ctx=8192. Dedupe
    # first: chunks overlap by design, so the same strong moment is scouted
    # twice and would otherwise burn two of the capped slots, crowding out
    # candidates that were only found once.
    cleanup_input = dedupe_moments(scouted_candidates)
    if len(cleanup_input) > cleanup_cap:
        cleanup_input = sorted(cleanup_input, key=ranking_value, reverse=True)[:cleanup_cap]
    if len(cleanup_input) < len(scouted_candidates):
        LOGGER.info(
            "pre-cleanup: %d scouted -> %d candidates (deduped, capped at %d)",
            len(scouted_candidates), len(cleanup_input), cleanup_cap,
        )

    cleaned_candidates = await cleanup_and_refine_candidates(
        cleanup_refine_provider,
        cleanup_input,
        requirements=requirements,
        prompt=cleanup_refine_prompt,
        temperature=cleanup_refine_temp,
        timeout=cleanup_refine_timeout,
        max_items=max(12, target_total * 3),
        json_retries=json_retries,
    )
    cleaned_candidates = _guard_stage_output(
        cleaned_candidates, cleanup_input, stage="cleanup", enabled=require_overlap,
    )
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
                noise_db=audio_noise_db,
                silence_min_s=audio_silence_min_s,
                timeout_s=audio_timeout_s,
            )
    atomic_write_json(outdir / "cleaned_candidates.json", build_candidate_json(cleaned_candidates))

    judged_candidates = await judge_candidates(
        judge_metadata_provider,
        cleaned_candidates,
        requirements=requirements,
        prompt=judge_metadata_prompt,
        temperature=judge_metadata_temp,
        timeout=judge_metadata_timeout,
        json_retries=json_retries,
        episode_context=episode_context,
        candidates_payload=(
            build_judge_payload(
                cleaned_candidates,
                index,
                max_candidates=judge_max_candidates,
                head_seconds=judge_head_s,
                tail_seconds=judge_tail_s,
                max_excerpt_chars=judge_excerpt_chars,
            )
            if judge_ctx_enabled
            else None
        ),
    )
    if not judged_candidates:
        judged_candidates = list(cleaned_candidates)
    judged_candidates = _guard_stage_output(
        judged_candidates, cleaned_candidates, stage="judge", enabled=require_overlap,
    )

    # Ground the quotes in the transcript, then anchor the bounds to real
    # speech boundaries. Order matters: quote refinement may widen a clip, and
    # snapping should act on the widened bounds.
    judged_candidates = apply_quote_verification(judged_candidates, index, **quote_conf)
    judged_candidates = snap_records(
        judged_candidates, index, enabled=snap_enabled, max_shift_s=snap_max_shift_s,
    )
    judged_candidates = clamp_records_to_episode(judged_candidates, float(duration))
    judged_candidates = annotate_speech_rate(judged_candidates, index)

    # Rank and finalize exactly once. Ranking twice used to feed the combined
    # priority back in as the next pass's base score.
    selected = rank_moments(
        judged_candidates,
        clip_type_quotas=quotas,
        scoring_weights=scoring_weights,
    )
    if judged_candidates and not selected:
        # Every candidate fell outside the configured quotas. Emitting them
        # unranked would quietly override the config, so report it instead.
        LOGGER.warning(
            "no candidate matched the configured clip quotas %s "
            "(%d candidates, types: %s)",
            quotas,
            len(judged_candidates),
            sorted({record.clip_type for record in judged_candidates}),
        )
    final_moments = finalize_moment_list(selected)
    final_payload = [moment.to_dict() for moment in final_moments]
    atomic_write_json(outdir / "moments.json", final_payload)
    (outdir / "reels.md").write_text(
        render_reels_summary_markdown(final_moments),
        encoding="utf-8",
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
    final = finalize_moment_list(ranked)
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
