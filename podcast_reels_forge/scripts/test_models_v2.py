#!/usr/bin/env python3
"""Test multiple Ollama models with simple direct prompt."""

import argparse
import json
import time
from pathlib import Path

import requests

MODELS = [
    "qwen3:latest",
    "deepseek-r1:8b",
    "gemma3:4b",
    "gemma2:9b",
    "gemini-3-flash-preview:latest",
]

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
TARGET_MOMENTS = 5
MAX_PASSES = 4

# Models that tend to drift into commentary; force JSON output when possible.
JSON_MODELS = {
    "gemma3:4b",
}


def load_transcript(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])


def segments_to_lines(segments: list[dict]) -> list[str]:
    lines: list[str] = []
    for seg in segments:
        start = int(seg.get("start", 0))
        end = int(seg.get("end", 0))
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(f"[{start}-{end}] {text}")
    return lines


def prepare_transcript_for_model(model: str, segments: list[dict]) -> tuple[str, set[int], set[int]]:
    """Return transcript text and allowed start/end timestamps for the portion included.

    Some models (notably qwen3/deepseek) can produce empty output on very long prompts.
    We keep them on a smaller window while still enabling 30–90s spans.
    """
    if model.startswith("gemini-"):
        max_chars = 45000
    elif model.startswith("qwen3"):
        max_chars = 8000
    elif model.startswith("qwen3") or model.startswith("deepseek-"):
        max_chars = 14000
    else:
        max_chars = 30000

    lines = segments_to_lines(segments)

    def clamp(n: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, n))

    # Determine which segment timestamps are actually in the prompt window.
    # For some models, sample the transcript across the timeline to avoid
    # returning 5 moments from the same ~30 seconds.
    if model.startswith("qwen3") or model.startswith("deepseek-"):
        n = len(segments)
        slice_size = 40 if model.startswith("qwen3") else 60
        mid = n // 2
        ranges = [
            (0, clamp(slice_size, 0, n)),
            (clamp(mid - slice_size // 2, 0, n), clamp(mid + slice_size // 2, 0, n)),
            (clamp(n - slice_size, 0, n), n),
        ]

        picked_idx: list[int] = []
        for a, b in ranges:
            picked_idx.extend(range(a, b))

        # De-dupe while preserving order
        seen: set[int] = set()
        picked_idx = [i for i in picked_idx if not (i in seen or seen.add(i))]

        out_lines = []
        used_segments: list[dict] = []
        total = 0
        for i in picked_idx:
            line = lines[i]
            total += len(line) + 1
            if total > max_chars:
                break
            out_lines.append(line)
            used_segments.append(segments[i])
    else:
        out_lines = []
        total = 0
        for line, seg in zip(lines, segments, strict=False):
            total += len(line) + 1
            if total > max_chars:
                break
            out_lines.append(line)
        used_segments = segments[: len(out_lines)]

    allowed_starts = {int(s.get("start", 0)) for s in used_segments}
    allowed_ends = {int(s.get("end", 0)) for s in used_segments}
    return "\n".join(out_lines), allowed_starts, allowed_ends


def get_prompt(model: str, transcript: str, *, already_selected: list[dict] | None = None, pass_index: int = 1) -> str:
    selected_ranges = ""
    if already_selected:
        ranges = []
        for m in already_selected:
            try:
                ranges.append(f"[{int(m.get('start'))}-{int(m.get('end'))}]")
            except (TypeError, ValueError):
                continue
        if ranges:
            selected_ranges = "\nУЖЕ ВЫБРАНО (НЕ ПОВТОРЯТЬ И НЕ ПЕРЕСЕКАТЬСЯ): " + ", ".join(ranges) + "\n"

    if model.startswith("gemini-"):
        return f"""Ты — редактор коротких вертикальных роликов.
Задача: по транскрипту подкаста выбрать 5 самых вирусных моментов для TikTok/Reels.

Формат транскрипта: [начало-конец] текст (секунды)

ТРАНСКРИПТ:
{transcript}

ТРЕБОВАНИЯ К ВЫВОДУ:
- Верни ТОЛЬКО JSON массив (без markdown/код-блоков/пояснений).
- Каждая строка должна быть валидным JSON (без переносов строк внутри строковых значений).
- Ровно 5 объектов.
- start/end должны совпадать с таймкодами из транскрипта.
- Длительность каждого момента: 30–90 секунд.
{selected_ranges}

Если ранее ты выдавал слишком короткие моменты (<30с), исправь это: выбирай диапазоны, покрывающие несколько реплик/фрагментов.

СХЕМА:
[
  {{"start": 145, "end": 211, "title": "Короткий хук", "quote": "Сильная цитата", "score": 9, "why": "Почему это репостят"}}
]
"""

    if model.startswith("qwen3"):
        return f"""Выбери ровно {TARGET_MOMENTS} вирусных моментов из транскрипта.

ТРАНСКРИПТ:
{transcript}

ОТВЕТ: верни ТОЛЬКО JSON-массив из {TARGET_MOMENTS} объектов (без текста/markdown).
Поля каждого объекта: start,end,title,quote,score,why.
Правила: start = левое число из [start-end], end = правое число из более позднего сегмента; end-start = 30..90.
{selected_ranges}
JSON:"""

    # Generic prompt for local models
    extra = ""
    if pass_index > 1:
        extra = (
            "\nВАЖНО: в прошлой попытке были слишком короткие клипы. "
            "Выбирай более длинные непрерывные диапазоны 30–90 секунд (не 3–10 секунд).\n"
        )

    return f"""Проанализируй транскрипт подкаста и найди {TARGET_MOMENTS} самых вирусных моментов для TikTok/Reels.

Формат транскрипта: [начало-конец] текст (время в секундах)

ТРАНСКРИПТ:
{transcript}

ЗАДАНИЕ: Выбери 5 моментов с наибольшим вирусным потенциалом.

Верни результат в формате JSON массива:
[
  {{"start": 145, "end": 200, "title": "Название момента", "quote": "Яркая цитата", "score": 9, "why": "Почему это вирусно"}}
]

Правила:
1. start должен быть РАВЕН числу начала одного из сегментов (левое число в [start-end])
2. end должен быть РАВЕН числу конца другого (более позднего) сегмента (правое число в [start-end])
    ВАЖНО: start и end НЕ обязаны быть из одной строки; можно (и нужно) объединять несколько сегментов.
    Пример: если есть [571-584] и [647-661], то валидный момент: start=571 end=661 (длительность 90с).
3a. Выбирай моменты из разных частей подкаста (ранний/середина/конец), не все подряд из одного места.
3. Момент длится 30-90 секунд (end-start в секундах)
4. score от 1 до 10
5. Не повторяй и не пересекайся с уже выбранными диапазонами
{extra}
{selected_ranges}

JSON:"""


def extract_json_array(text: str) -> list:
    """Extract JSON array from response."""
    import re
    
    # Remove think tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text)
    
    # First: try strict JSON parse (helps when Ollama json-mode returns an object)
    stripped = text.strip()
    if stripped:
        try:
            obj = json.loads(stripped)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for key in ("moments", "clips", "choices", "items", "objects", "results", "segments"):
                    val = obj.get(key)
                    if isinstance(val, list):
                        return val
        except Exception:
            pass

    # Fallback: find first JSON array by bracket matching
    start = text.find("[")
    if start == -1:
        return []
    
    depth = 0
    end = start
    for i, c in enumerate(text[start:], start):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    
    if end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return []


def is_duration_valid(m: dict, allowed_starts: set[int], allowed_ends: set[int]) -> bool:
    try:
        start = int(m.get("start"))
        end = int(m.get("end"))
    except (TypeError, ValueError):
        return False
    if start not in allowed_starts:
        return False
    if end not in allowed_ends:
        return False
    duration = end - start
    return 30 <= duration <= 90


def snap_to_allowed(value: int, allowed: set[int], *, max_delta: int = 10) -> int | None:
    """Snap a timestamp to the closest allowed value within max_delta seconds."""
    best: int | None = None
    best_dist = max_delta + 1
    for a in allowed:
        dist = abs(a - value)
        if dist <= max_delta and dist < best_dist:
            best = a
            best_dist = dist
            if best_dist == 0:
                break
    return best


def normalize_timestamps(m: dict, allowed_starts: set[int], allowed_ends: set[int]) -> dict:
    """Snap start/end to nearby segment boundaries when the model is slightly off."""
    try:
        start = int(m.get("start"))
        end = int(m.get("end"))
    except (TypeError, ValueError):
        return m

    snapped_start = snap_to_allowed(start, allowed_starts, max_delta=10)
    snapped_end = snap_to_allowed(end, allowed_ends, max_delta=10)
    if snapped_start is None and snapped_end is None:
        return m

    out = dict(m)
    if snapped_start is not None:
        out["start"] = snapped_start
    if snapped_end is not None:
        out["end"] = snapped_end
    return out


def ranges_overlap(a: dict, b: dict) -> bool:
    try:
        a0, a1 = int(a.get("start")), int(a.get("end"))
        b0, b1 = int(b.get("start")), int(b.get("end"))
    except (TypeError, ValueError):
        return False
    return not (a1 <= b0 or b1 <= a0)


def normalize_duration(m: dict, allowed_ends: set[int]) -> dict:
    """Snap end to the nearest allowed end so duration falls into 30–90s.

    Helps models that keep choosing a single short segment.
    """
    try:
        start = int(m.get("start"))
        end = int(m.get("end"))
    except (TypeError, ValueError):
        return m

    duration = end - start
    if 30 <= duration <= 90:
        return m

    lo = start + 30
    hi = start + 90
    candidates = [e for e in allowed_ends if lo <= e <= hi]
    if not candidates:
        return m

    out = dict(m)
    out["end"] = min(candidates)
    return out


def dedupe_moments(
    existing: list[dict],
    incoming: list[dict],
    allowed_starts: set[int],
    allowed_ends: set[int],
    *,
    allow_overlap: bool,
) -> list[dict]:
    out = list(existing)
    for m in incoming:
        if not isinstance(m, dict):
            continue
        if not ("start" in m and "end" in m):
            continue
        m = normalize_timestamps(m, allowed_starts, allowed_ends)
        m = normalize_duration(m, allowed_ends)
        if not is_duration_valid(m, allowed_starts, allowed_ends):
            continue
        # Always avoid exact duplicates
        try:
            s = int(m.get("start"))
            e = int(m.get("end"))
            if any(int(prev.get("start")) == s and int(prev.get("end")) == e for prev in out):
                continue
        except Exception:
            pass
        if not allow_overlap:
            if any(ranges_overlap(m, prev) for prev in out):
                continue
        out.append(m)
        if len(out) >= TARGET_MOMENTS:
            break
    return out


def run_model_iterative(model: str, transcript: str, allowed_starts: set[int], allowed_ends: set[int]) -> tuple[list[dict], float, str]:
    print(f"  {model}...", end=" ", flush=True)
    start_time = time.time()
    all_raw_parts: list[str] = []
    selected: list[dict] = []

    for pass_index in range(1, MAX_PASSES + 1):
        prompt = get_prompt(model, transcript, already_selected=selected, pass_index=pass_index)
        try:
            if model.startswith("qwen3"):
                system = "Верни только JSON. Без текста."
                resp = requests.post(
                    OLLAMA_CHAT_URL,
                    json={
                        "model": model,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "options": {"temperature": 0.0, "num_predict": 1400},
                    },
                    timeout=900,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("error"):
                    all_raw_parts.append(f"\n\n--- PASS {pass_index} ERROR_FIELD ---\n\n{data.get('error')}")
                    break
                msg = data.get("message") or {}
                raw = msg.get("content") or ""
                if (not raw) and isinstance(msg.get("thinking"), str):
                    raw = msg.get("thinking", "")
            else:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2 if model.startswith("gemini-") else 0.3,
                        "num_predict": 4096 if model.startswith("gemini-") else 2048,
                    },
                }
                if model in JSON_MODELS:
                    payload["format"] = "json"

                resp = requests.post(
                    OLLAMA_URL,
                    json=payload,
                    timeout=300 if model.startswith("gemini-") else 900,
                )

                resp.raise_for_status()
                data = resp.json()
                if data.get("error"):
                    all_raw_parts.append(f"\n\n--- PASS {pass_index} ERROR_FIELD ---\n\n{data.get('error')}")
                    break

                raw = data.get("response", "")
                if (not raw) and isinstance(data.get("thinking"), str):
                    raw = data.get("thinking", "")

            # end qwen3 vs generate split

            all_raw_parts.append(f"\n\n--- PASS {pass_index} ---\n\n{raw}")

            moments = extract_json_array(raw)
            candidates = [m for m in moments if isinstance(m, dict) and "start" in m and "end" in m]
            allow_overlap = model.startswith("qwen3") or model.startswith("deepseek-")
            selected = dedupe_moments(selected, candidates, allowed_starts, allowed_ends, allow_overlap=allow_overlap)

            if len(selected) >= TARGET_MOMENTS:
                break

        except Exception as e:
            all_raw_parts.append(f"\n\n--- PASS {pass_index} ERROR ---\n\n{e}")
            break

    elapsed = time.time() - start_time
    print(f"✓ {len(selected)}/{TARGET_MOMENTS} valid, {elapsed:.0f}s")
    return selected, elapsed, "".join(all_raw_parts).lstrip()


def avg_score(moments: list[dict]) -> float:
    def norm(v: float) -> float:
        # Many prompts expect score 1..10, but some models output 0..100.
        # Normalize for comparison/reporting only.
        if v <= 0:
            return 0.0
        if 0 < v <= 1:
            v = v * 10
        elif v > 10:
            v = v / 10
        return float(max(0.0, min(10.0, v)))

    scores = [norm(float(m.get("score"))) for m in moments if isinstance(m.get("score"), (int, float))]
    return (sum(scores) / len(scores)) if scores else 0.0


def generate_report(output_dir: Path, models: list[str]) -> None:
    report_lines = ["# Model Comparison v2\n"]
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
    report_lines.append("| Model | Time | Moments | Avg Score |")
    report_lines.append("|-------|------|---------|-----------|")

    best_model: str | None = None
    best_tuple: tuple[int, float, float] | None = None

    loaded: dict[str, dict] = {}
    for model in models:
        safe_name = model.replace(":", "_").replace("/", "_")
        p = output_dir / f"{safe_name}_v2.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        loaded[model] = data

    for model in models:
        data = loaded.get(model)
        if not data:
            continue
        moments = data.get("moments", [])
        elapsed = data.get("time", 0)
        avg = avg_score(moments)
        report_lines.append(f"| {model} | {elapsed:.0f}s | {len(moments)} | {avg:.1f} |")

        current = (len(moments), avg, -float(elapsed))
        if best_tuple is None or current > best_tuple:
            best_tuple = current
            best_model = model

    if best_model and best_model in loaded:
        report_lines.append(f"\n## Winner: {best_model}\n")
        report_lines.append("### Best moments:\n")
        for i, m in enumerate(loaded[best_model].get("moments", [])[:TARGET_MOMENTS], 1):
            report_lines.append(f"{i}. **{m.get('title', '?')}** [{m.get('start')}-{m.get('end')}s]")
            score = m.get("score")
            if isinstance(score, (int, float)):
                report_lines.append(f"   - Score: {avg_score([{'score': score}]):.1f}/10")
            else:
                report_lines.append(f"   - Score: {score}")
            report_lines.append(f"   - Quote: {str(m.get('quote', '?'))[:80]}")
            report_lines.append(f"   - Why: {str(m.get('why', '?'))[:80]}")
            report_lines.append("")

    (output_dir / "model_comparison_v2.md").write_text("\n".join(report_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple Ollama models.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of models to run (others will be kept from existing v2 JSON outputs).",
    )
    args = parser.parse_args()

    transcript_path = Path("/home/borm/VibeCoding/Podcast Reels Forge/output/audio.json")
    output_dir = Path("/home/borm/VibeCoding/Podcast Reels Forge/anal")
    
    print("=" * 50)
    print("Simple Model Test")
    print("=" * 50)
    
    segments = load_transcript(str(transcript_path))
    sample_transcript, _, _ = prepare_transcript_for_model(MODELS[0], segments)
    sample_prompt = get_prompt(MODELS[0], sample_transcript)
    print(f"Transcript: {len(segments)} segments, sample prompt: {len(sample_prompt)} chars\n")
    
    models_to_run = args.models if args.models else MODELS
    results: dict[str, dict] = {}

    for model in models_to_run:
        transcript, allowed_starts, allowed_ends = prepare_transcript_for_model(model, segments)
        moments, elapsed, raw = run_model_iterative(model, transcript, allowed_starts, allowed_ends)
        results[model] = {"moments": moments, "time": elapsed, "raw": raw}

        safe_name = model.replace(":", "_").replace("/", "_")
        (output_dir / f"{safe_name}_v2.json").write_text(
            json.dumps({"model": model, "time": elapsed, "moments": moments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{safe_name}_v2_raw.txt").write_text(raw, encoding="utf-8")

    # Print quick summary for the models we just ran
    print("\n" + "=" * 50)
    print("RESULTS (this run):")
    print("-" * 50)
    for model, data in results.items():
        moments = data["moments"]
        elapsed = data["time"]
        avg = avg_score(moments)
        print(f"{model}: time {elapsed:.0f}s, moments {len(moments)}, avg {avg:.1f}")

    # Always regenerate report from existing v2 JSON files
    generate_report(output_dir, MODELS)
    print(f"\nReport: {output_dir / 'model_comparison_v2.md'}")


if __name__ == "__main__":
    main()
