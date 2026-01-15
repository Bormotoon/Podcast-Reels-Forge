#!/usr/bin/env python3
"""Test multiple Ollama models for viral moment detection quality."""

import json
import time
from pathlib import Path

import requests

# Models to test (excluding vision models and too small ones)
MODELS = [
    "gemma2:9b",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen3:latest",
    "deepseek-r1:8b",
    "gemma3:12b-it-qat",
]

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TIMEOUT = 600

# Universal prompt optimized for all models
PROMPT_TEMPLATE = """<|im_start|>system
Ты JSON-генератор. Ты анализируешь транскрипты подкастов и возвращаешь ТОЛЬКО валидный JSON без пояснений.
<|im_end|>
<|im_start|>user
Найди 5 самых вирусных моментов из транскрипта подкаста.

Формат транскрипта: [секунда_начала-секунда_конца] текст

Критерии вирусности (score 8-10):
- Эмоция (смех, шок, удивление)
- Конфликт или противоречие  
- Полезный совет или инсайт
- Цитатная фраза

Правила:
- start и end берутся ТОЧНО из временных меток транскрипта
- Длительность момента: 30-90 секунд
- Момент должен начинаться с сильной фразы
- Момент должен заканчиваться логично

ТРАНСКРИПТ:
{transcript}

Верни ответ СТРОГО в формате JSON (без markdown, без пояснений):
<|im_end|>
<|im_start|>assistant
{{"moments": ["""


def load_transcript(path: str) -> tuple[list[dict], float]:
    """Load transcript from JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", []), data.get("duration", 0)


def segments_to_text(segments: list[dict], max_chars: int = 15000) -> str:
    """Convert segments to compact text format."""
    lines = []
    for seg in segments:
        start = int(seg.get("start", 0))
        end = int(seg.get("end", 0))
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{start}-{end}] {text}")
    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars]
        last_nl = result.rfind("\n")
        if last_nl > max_chars * 0.8:
            result = result[:last_nl]
    return result


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    import re
    
    # Prepend the partial JSON we started with if needed
    if not text.strip().startswith("{"):
        text = '{"moments": [' + text
    
    # Try to find complete JSON
    # First, try to complete the JSON if it was cut off
    if text.count("{") > text.count("}"):
        # Add missing closing braces
        text = text.rstrip()
        while text.count("{") > text.count("}"):
            text += "}"
        while text.count("[") > text.count("]"):
            text += "]"
    
    # Try to find JSON in markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object
    start = text.find("{")
    if start >= 0:
        # Find the last valid closing brace
        depth = 0
        end = start
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        
        if end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    
    raise ValueError("No JSON found")


def run_model(model: str, prompt: str) -> tuple[dict, float, str]:
    """Run a single model and return result, time, and raw response."""
    print(f"  Testing {model}...", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": 4096},
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        
        raw = response.json().get("response", "")
        elapsed = time.time() - start_time
        
        try:
            result = extract_json(raw)
            moments = result.get("moments", [])
            print(f"✓ {len(moments)} moments, {elapsed:.1f}s")
            return result, elapsed, raw
        except (json.JSONDecodeError, ValueError) as e:
            print(f"✗ JSON error: {e}")
            return {"error": str(e), "raw": raw[:500]}, elapsed, raw
            
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        print(f"✗ Request error: {e}")
        return {"error": str(e)}, elapsed, ""


def analyze_results(results: dict) -> str:
    """Analyze results and generate markdown report."""
    lines = ["# Анализ моделей для поиска вирусных моментов\n"]
    lines.append(f"**Дата тестирования:** {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("## Сводная таблица\n")
    lines.append("| Модель | Время | Моменты | JSON OK | Средний Score |")
    lines.append("|--------|-------|---------|---------|---------------|")
    
    scores = {}
    for model, data in results.items():
        result = data.get("result", {})
        elapsed = data.get("time", 0)
        moments = result.get("moments", [])
        
        # Filter out invalid moments (strings instead of dicts)
        moments = [m for m in moments if isinstance(m, dict)]
        
        json_ok = "✓" if moments and "error" not in result else "✗"
        avg_score = 0
        if moments:
            scores_list = [m.get("score", 0) for m in moments if isinstance(m.get("score"), (int, float))]
            avg_score = sum(scores_list) / len(scores_list) if scores_list else 0
        
        scores[model] = {
            "time": elapsed,
            "count": len(moments),
            "json_ok": json_ok == "✓",
            "avg_score": avg_score,
            "moments": moments,
        }
        
        lines.append(f"| {model} | {elapsed:.1f}s | {len(moments)} | {json_ok} | {avg_score:.1f} |")
    
    lines.append("\n## Детальный анализ по моделям\n")
    
    for model, data in results.items():
        lines.append(f"### {model}\n")
        result = data.get("result", {})
        elapsed = data.get("time", 0)
        
        if "error" in result:
            lines.append(f"**Ошибка:** {result['error']}\n")
            if "raw" in result:
                lines.append(f"```\n{result['raw'][:300]}...\n```\n")
            continue
        
        moments = result.get("moments", [])
        lines.append(f"- **Время генерации:** {elapsed:.1f}s")
        lines.append(f"- **Найдено моментов:** {len(moments)}\n")
        
        if moments:
            lines.append("**Найденные моменты:**\n")
            for i, m in enumerate(moments[:5], 1):
                start = m.get("start", "?")
                end = m.get("end", "?")
                title = m.get("title", "Без названия")
                score = m.get("score", "?")
                why = m.get("why", "")
                quote = m.get("quote", "")
                
                lines.append(f"{i}. **{title}** [{start}-{end}s] (score: {score})")
                if quote:
                    lines.append(f"   > {quote[:100]}...")
                if why:
                    lines.append(f"   - Почему: {why[:100]}...")
                lines.append("")
    
    # Recommendations
    lines.append("\n## Рекомендации\n")
    
    # Find best model
    valid_models = {k: v for k, v in scores.items() if v["json_ok"] and v["count"] >= 3}
    
    if valid_models:
        # Score by: JSON quality, number of moments, avg score, speed
        def model_score(m):
            s = scores[m]
            return (
                s["count"] * 2 +  # More moments is better
                s["avg_score"] * 1.5 +  # Higher scores better
                (100 / max(s["time"], 1)) * 0.5  # Faster is better
            )
        
        best = max(valid_models.keys(), key=model_score)
        fastest = min(valid_models.keys(), key=lambda m: scores[m]["time"])
        highest_score = max(valid_models.keys(), key=lambda m: scores[m]["avg_score"])
        
        lines.append(f"### 🏆 Лучшая модель: **{best}**\n")
        lines.append("- Оптимальный баланс качества и скорости\n")
        lines.append(f"### ⚡ Самая быстрая: **{fastest}** ({scores[fastest]['time']:.1f}s)\n")
        lines.append(f"### 🎯 Лучшие оценки: **{highest_score}** (avg: {scores[highest_score]['avg_score']:.1f})\n")
        
        lines.append("\n### Итоговая рекомендация\n")
        lines.append(f"Для production использования рекомендуется **{best}**.\n")
        
        # Quality analysis
        best_moments = scores[best]["moments"]
        if best_moments:
            lines.append("\n### Лучшие моменты от победителя:\n")
            for i, m in enumerate(sorted(best_moments, key=lambda x: x.get("score", 0), reverse=True)[:3], 1):
                lines.append(f"{i}. **{m.get('title', '?')}** [{m.get('start')}-{m.get('end')}s]")
                lines.append(f"   - Score: {m.get('score')}")
                lines.append(f"   - {m.get('why', '')[:150]}")
                lines.append("")
    else:
        lines.append("⚠️ Ни одна модель не выдала корректный JSON с достаточным количеством моментов.\n")
        lines.append("Рекомендуется проверить промпты или попробовать другие модели.\n")
    
    return "\n".join(lines)


def main():
    transcript_path = Path("/home/borm/VibeCoding/Podcast Reels Forge/output/audio.json")
    output_dir = Path("/home/borm/VibeCoding/Podcast Reels Forge/anal")
    
    print("=" * 60)
    print("Model Comparison Test for Viral Moment Detection")
    print("=" * 60)
    
    # Load transcript
    print(f"\nLoading transcript from {transcript_path}...")
    segments, duration = load_transcript(str(transcript_path))
    print(f"  Loaded {len(segments)} segments, duration: {duration:.0f}s")
    
    # Prepare prompt with first ~15000 chars of transcript
    transcript_text = segments_to_text(segments, max_chars=15000)
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)
    print(f"  Prompt length: {len(prompt)} chars")
    
    # Test each model
    print(f"\nTesting {len(MODELS)} models...")
    results = {}
    
    for model in MODELS:
        result, elapsed, raw = run_model(model, prompt)
        results[model] = {"result": result, "time": elapsed, "raw": raw}
        
        # Save individual result
        model_safe = model.replace(":", "_").replace("/", "_")
        out_file = output_dir / f"{model_safe}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"model": model, "time": elapsed, "result": result}, f, ensure_ascii=False, indent=2)
        
        # Also save raw response
        raw_file = output_dir / f"{model_safe}_raw.txt"
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(raw)
    
    # Generate report
    print("\nGenerating report...")
    report = analyze_results(results)
    
    report_path = output_dir / "model_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n✓ Report saved to {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
