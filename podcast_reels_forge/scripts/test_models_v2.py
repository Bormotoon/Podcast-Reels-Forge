#!/usr/bin/env python3
"""Gemma-only Ollama benchmark for transcript moment extraction."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

MODELS = [
    "gemma4:e4b",
    "gemma3:4b",
    "gemma3:12b",
    "gemma4:26b",
]

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TARGET_MOMENTS = 5
MAX_PASSES = 3
MAX_PROMPT_CHARS = 18000
DEFAULT_TRANSCRIPT = Path(__file__).resolve().parents[2] / "output" / "audio.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "anal"


def load_transcript(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not isinstance(segments, list):
        return []
    return [segment for segment in segments if isinstance(segment, dict)]


def valid_segment_rows(segments: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for seg in segments:
        try:
            start = int(float(seg.get("start", 0)))
            end = int(float(seg.get("end", 0)))
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text", "")).strip()
        if text:
            rows.append((f"[{start}-{end}] {text}", seg))
    return rows


def prepare_transcript(segments: list[dict[str, Any]]) -> tuple[str, set[int], set[int]]:
    rows = valid_segment_rows(segments)
    if not rows:
        return "", set(), set()

    lines = [line for line, _segment in rows]
    total_len = sum(len(line) + 1 for line in lines)
    if total_len <= MAX_PROMPT_CHARS:
        allowed_starts = {int(float(seg.get("start", 0))) for _line, seg in rows}
        allowed_ends = {int(float(seg.get("end", 0))) for _line, seg in rows}
        return "\n".join(lines), allowed_starts, allowed_ends

    n = len(lines)
    window = max(1, n // 4)
    ranges = [
        (0, min(n, window)),
        (max(0, n // 2 - window // 2), min(n, n // 2 + window // 2)),
        (max(0, n - window), n),
    ]

    picked_indices: list[int] = []
    for start, end in ranges:
        picked_indices.extend(range(start, end))

    seen: set[int] = set()
    sampled_lines: list[str] = []
    used_segments: list[dict[str, Any]] = []
    total = 0
    for idx in picked_indices:
        if idx in seen:
            continue
        seen.add(idx)
        line = lines[idx]
        projected = total + len(line) + 1
        if projected > MAX_PROMPT_CHARS:
            break
        sampled_lines.append(line)
        used_segments.append(rows[idx][1])
        total = projected

    allowed_starts = {int(float(s.get("start", 0))) for s in used_segments}
    allowed_ends = {int(float(s.get("end", 0))) for s in used_segments}
    return "\n".join(sampled_lines), allowed_starts, allowed_ends


def build_prompt(
    transcript: str,
    *,
    already_selected: list[dict[str, Any]] | None = None,
    pass_index: int = 1,
) -> str:
    selected_ranges = ""
    if already_selected:
        ranges = []
        for item in already_selected:
            try:
                ranges.append(f"[{int(item.get('start', -1))}-{int(item.get('end', -1))}]")
            except (TypeError, ValueError):
                continue
        if ranges:
            selected_ranges = "\nУЖЕ ВЫБРАНО (НЕ ПОВТОРЯТЬ И НЕ ПЕРЕСЕКАТЬСЯ): " + ", ".join(ranges)

    extra = ""
    if pass_index > 1:
        extra = (
            "\nЕсли в прошлой попытке моменты были слишком короткими, "
            "выбирай более длинные непрерывные диапазоны 30-90 секунд."
        )

    return f"""Ты - редактор коротких вертикальных роликов.
По транскрипту подкаста выбери 5 самых сильных моментов для Reels/TikTok.

Формат транскрипта: [начало-конец] текст

ТРАНСКРИПТ:
{transcript}

ТРЕБОВАНИЯ:
- Верни только JSON-массив, без markdown, пояснений и code block.
- Ровно 5 объектов.
- Каждый объект должен содержать поля: start, end, title, quote, score, why.
- start и end должны совпадать с таймкодами из транскрипта.
- Длительность каждого момента: 30-90 секунд.
- Не повторяй уже выбранные диапазоны.
{selected_ranges}
{extra}

JSON:"""


def extract_json_array(text: str) -> list[Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned)
    cleaned = cleaned.replace("```", "").strip()

    if cleaned:
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for key in ("moments", "clips", "choices", "items", "objects", "results", "segments"):
                    value = obj.get(key)
                    if isinstance(value, list):
                        return value
        except Exception:
            pass

    start = cleaned.find("[")
    if start == -1:
        return []

    depth = 0
    end = -1
    for idx, ch in enumerate(cleaned[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end > start:
        try:
            parsed = json.loads(cleaned[start:end])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def snap_to_allowed(value: int, allowed: set[int], *, max_delta: int = 10) -> int | None:
    best: int | None = None
    best_dist = max_delta + 1
    for candidate in allowed:
        dist = abs(candidate - value)
        if dist <= max_delta and dist < best_dist:
            best = candidate
            best_dist = dist
            if best_dist == 0:
                break
    return best


def normalize_moment(moment: dict[str, Any], allowed_starts: set[int], allowed_ends: set[int]) -> dict[str, Any]:
    try:
        start = int(moment.get("start", -1))
        end = int(moment.get("end", -1))
    except (TypeError, ValueError):
        return moment

    out = dict(moment)
    snapped_start = snap_to_allowed(start, allowed_starts)
    snapped_end = snap_to_allowed(end, allowed_ends)
    if snapped_start is not None:
        out["start"] = snapped_start
    if snapped_end is not None:
        out["end"] = snapped_end
    return out


def normalize_duration(moment: dict[str, Any], allowed_ends: set[int]) -> dict[str, Any]:
    try:
        start = int(moment.get("start", -1))
        end = int(moment.get("end", -1))
    except (TypeError, ValueError):
        return moment

    duration = end - start
    if 30 <= duration <= 90:
        return moment

    lo = start + 30
    hi = start + 90
    candidates = [candidate for candidate in allowed_ends if lo <= candidate <= hi]
    if not candidates:
        return moment

    out = dict(moment)
    out["end"] = min(candidates)
    return out


def ranges_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    try:
        a0, a1 = int(a.get("start", -1)), int(a.get("end", -1))
        b0, b1 = int(b.get("start", -1)), int(b.get("end", -1))
    except (TypeError, ValueError):
        return False
    return not (a1 <= b0 or b1 <= a0)


def is_duration_valid(moment: dict[str, Any], allowed_starts: set[int], allowed_ends: set[int]) -> bool:
    try:
        start = int(moment.get("start", -1))
        end = int(moment.get("end", -1))
    except (TypeError, ValueError):
        return False
    if start not in allowed_starts:
        return False
    if end not in allowed_ends:
        return False
    return 30 <= (end - start) <= 90


def dedupe_moments(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    allowed_starts: set[int],
    allowed_ends: set[int],
) -> list[dict[str, Any]]:
    out = list(existing)
    for moment in incoming:
        if not isinstance(moment, dict):
            continue
        if "start" not in moment or "end" not in moment:
            continue
        normalized = normalize_moment(moment, allowed_starts, allowed_ends)
        normalized = normalize_duration(normalized, allowed_ends)
        if not is_duration_valid(normalized, allowed_starts, allowed_ends):
            continue

        try:
            start = int(normalized.get("start", -1))
            end = int(normalized.get("end", -1))
            if any(int(prev.get("start", -1)) == start and int(prev.get("end", -1)) == end for prev in out):
                continue
        except Exception:
            pass

        if any(ranges_overlap(normalized, prev) for prev in out):
            continue

        out.append(normalized)
        if len(out) >= TARGET_MOMENTS:
            break
    return out


def run_model_iterative(
    model: str,
    transcript: str,
    allowed_starts: set[int],
    allowed_ends: set[int],
) -> tuple[list[dict[str, Any]], float, str]:
    print(f"  {model}...", end=" ", flush=True)
    started_at = time.time()
    raw_parts: list[str] = []
    selected: list[dict[str, Any]] = []

    for pass_index in range(1, MAX_PASSES + 1):
        prompt = build_prompt(transcript, already_selected=selected, pass_index=pass_index)
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 2048,
                    },
                    "format": "json",
                },
                timeout=900,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raw_parts.append(f"\n\n--- PASS {pass_index} ERROR_FIELD ---\n\n{payload.get('error')}")
                break

            raw = payload.get("response", "")
            if not raw and isinstance(payload.get("thinking"), str):
                raw = payload["thinking"]

            raw_parts.append(f"\n\n--- PASS {pass_index} ---\n\n{raw}")

            moments = extract_json_array(raw)
            candidates = [moment for moment in moments if isinstance(moment, dict)]
            selected = dedupe_moments(selected, candidates, allowed_starts, allowed_ends)
            if len(selected) >= TARGET_MOMENTS:
                break
        except Exception as exc:
            raw_parts.append(f"\n\n--- PASS {pass_index} ERROR ---\n\n{exc}")
            break

    elapsed = time.time() - started_at
    print(f"✓ {len(selected)}/{TARGET_MOMENTS} valid, {elapsed:.0f}s")
    return selected, elapsed, "".join(raw_parts).lstrip()


def avg_score(moments: list[dict[str, Any]]) -> float:
    scores = []
    for moment in moments:
        score = moment.get("score")
        if not isinstance(score, (int, float)):
            continue
        value = float(score)
        if value <= 0:
            continue
        if value <= 1:
            value *= 10
        elif value > 10:
            value /= 10
        scores.append(max(0.0, min(10.0, value)))
    return (sum(scores) / len(scores)) if scores else 0.0


def generate_report(output_dir: Path, models: list[str]) -> None:
    report_lines = ["# Model Comparison v2", ""]
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("")
    report_lines.append("| Model | Time | Moments | Avg Score |")
    report_lines.append("|-------|------|---------|-----------|")

    loaded: dict[str, dict[str, Any]] = {}
    for model in models:
        safe_name = model.replace(":", "_").replace("/", "_")
        path = output_dir / f"{safe_name}_v2.json"
        if not path.exists():
            continue
        loaded[model] = json.loads(path.read_text(encoding="utf-8"))

    best_model: str | None = None
    best_tuple: tuple[int, float, float] | None = None
    for model in models:
        data = loaded.get(model)
        if not data:
            continue
        moments = data.get("moments", [])
        moment_list = moments if isinstance(moments, list) else []
        elapsed = float(data.get("time", 0))
        avg = avg_score(moment_list)
        report_lines.append(f"| {model} | {elapsed:.0f}s | {len(moment_list)} | {avg:.1f} |")

        current = (len(moment_list), avg, -elapsed)
        if best_tuple is None or current > best_tuple:
            best_tuple = current
            best_model = model

    if best_model and best_model in loaded:
        report_lines.extend(["", f"## Winner: {best_model}", "", "### Best moments:", ""])
        moments = loaded[best_model].get("moments", [])
        if isinstance(moments, list):
            for index, moment in enumerate(moments[:TARGET_MOMENTS], 1):
                if not isinstance(moment, dict):
                    continue
                report_lines.append(
                    f"{index}. **{moment.get('title', '?')}** [{moment.get('start')}-{moment.get('end')}s]",
                )
                score = moment.get("score")
                if isinstance(score, (int, float)):
                    report_lines.append(f"   - Score: {avg_score([{'score': score}]):.1f}/10")
                else:
                    report_lines.append(f"   - Score: {score}")
                report_lines.append(f"   - Quote: {str(moment.get('quote', '?'))[:80]}")
                report_lines.append(f"   - Why: {str(moment.get('why', '?'))[:80]}")
                report_lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_comparison_v2.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Gemma Ollama models.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model list. Defaults to the Gemma role lineup.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_TRANSCRIPT,
        help="Path to a transcript JSON file with a segments list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark outputs and the comparison report.",
    )
    args = parser.parse_args()

    transcript_path = args.transcript
    if not transcript_path.exists():
        raise SystemExit(f"Transcript not found: {transcript_path}")

    print("=" * 50)
    print("Gemma Model Test")
    print("=" * 50)

    segments = load_transcript(transcript_path)
    transcript, allowed_starts, allowed_ends = prepare_transcript(segments)
    if not transcript:
        raise SystemExit("Transcript is empty")

    models_to_run = args.models if args.models else MODELS
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for model in models_to_run:
        model_transcript = transcript
        moments, elapsed, raw = run_model_iterative(model, model_transcript, allowed_starts, allowed_ends)
        results[model] = {"moments": moments, "time": elapsed, "raw": raw}

        safe_name = model.replace(":", "_").replace("/", "_")
        (output_dir / f"{safe_name}_v2.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "time": elapsed,
                    "moments": moments,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / f"{safe_name}_v2_raw.txt").write_text(raw, encoding="utf-8")

    print("\n" + "=" * 50)
    print("RESULTS (this run):")
    print("-" * 50)
    for model, data in results.items():
        moments = data["moments"]
        elapsed = data["time"]
        avg = avg_score(moments)
        print(f"{model}: time {elapsed:.0f}s, moments {len(moments)}, avg {avg:.1f}")

    generate_report(output_dir, models_to_run)
    print(f"\nReport: {output_dir / 'model_comparison_v2.md'}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        import os
        import sys

        print(f"Testing failed: {exc}", file=sys.stderr)
        if os.environ.get("DEBUG_FORGE") == "1":
            import traceback

            traceback.print_exc()
        raise SystemExit(1)
