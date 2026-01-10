# Руководство по промптам / Prompts Guide

## Структура промптов

Промпты находятся в папке `prompts/` и организованы по языкам:

Prompts live in the `prompts/` directory and are organized by language:

```text
prompts/
├── ru/
│   ├── chunk_default.txt    # Анализ чанка (основной)
│   ├── chunk_a.txt          # Вариант A
│   ├── chunk_b.txt          # Вариант B
│   ├── select_default.txt   # Выбор моментов (основной)
│   ├── select_a.txt         # Вариант A
│   └── select_b.txt         # Вариант B
└── en/
    ├── chunk_default.txt
    ├── chunk_a.txt
    ├── chunk_b.txt
    ├── select_default.txt
    ├── select_a.txt
    └── select_b.txt
```

## Типы промптов

### chunk_*.txt — Анализ чанка

Этот промпт используется для анализа каждого временного сегмента транскрипта.

This prompt is used to analyze each time chunk of the transcript.

**Доступные переменные:**

**Ожидаемый формат ответа:**

Expected response format:

```json
{
  "moment": {
    "start": 123.45,
    "end": 178.90,
    "title": "Название момента / Moment title",
    "quote": "Ключевая цитата / Key quote",
    "why": "Почему это виральный момент / Why it's viral"
  }
}
```

### select_*.txt — Финальный выбор

Этот промпт используется для выбора лучших моментов из всех найденных кандидатов.

This prompt selects the best moments from all candidates.

**Доступные переменные:**

**Ожидаемый формат ответа:**

Expected response format:

```json
{
  "moments": [
    {
      "start": 123.45,
      "end": 178.90,
      "title": "Название / Title",
      "quote": "Цитата / Quote",
      "why": "Причина / Why",
      "score": 0.95,
      "hook": "Крючок / Hook",
      "caption": "Подпись / Caption",
      "hashtags": ["#подкаст/#podcast", "#интервью/#interview"]
    }
  ]
}
```

## A/B тестирование

A/B testing

### Запуск оценки

Running evaluation

```bash
python -m podcast_reels_forge.scripts.evaluate_prompts \
    --transcript output/video.json \
    --outdir output \
    --variants default,a,b \
    --provider ollama \
    --model gemma2:9b
```

### Формат результатов

Results format

Файл `output/prompt_eval.json`:

```json
{
  "transcript": "output/video.json",
  "provider": "ollama",
  "model": "gemma2:9b",
  "variants": [
    {
      "variant": "default",
      "moments": 4,
      "avg_score": 0.85,
      "avg_duration": 45.2,
      "violations": 0
    },
    {
      "variant": "a",
      "moments": 4,
      "avg_score": 0.82,
      "avg_duration": 38.7,
      "violations": 1
    }
  ],
  "stability_jaccard": {
    "default": {"default": 1.0, "a": 0.6, "b": 0.5},
    "a": {"default": 0.6, "a": 1.0, "b": 0.7}
  },
  "best_variant": "default"
}
```

### Метрики

## Советы по созданию промптов

Tips for writing prompts

### Для chunk промптов

For chunk prompts

1. **Будьте конкретны** — укажите, что именно считается "виральным"
2. **Задайте критерии** — эмоциональность, юмор, инсайты
3. **Укажите формат** — точный JSON формат ответа
4. **Ограничьте длину** — напомните про r_min/r_max

### Для select промптов

For select prompts

1. **Приоритизация** — укажите критерии ранжирования
2. **Разнообразие** — попросите выбирать разные типы моментов
3. **SMM метаданные** — попросите генерировать hook, caption, hashtags
4. **Score** — определите критерии для score (0-1)

## Примеры промптов

Prompt examples

### chunk_default.txt (RU)

```text
Ты — эксперт по созданию вирусного контента для социальных сетей.

Проанализируй следующий фрагмент транскрипта подкаста и найди ОДИН самый
интересный момент длительностью от {r_min} до {r_max} секунд.

Критерии виральности:
- Эмоциональные высказывания
- Неожиданные инсайты или признания
- Юмор или ирония
- Спорные или провокационные мнения
- Истории из жизни

Транскрипт (формат: [начало-конец] текст):
{transcript}

Ответь ТОЛЬКО валидным JSON:
{
  "moment": {
    "start": <число>,
    "end": <число>,
    "title": "<короткий заголовок>",
    "quote": "<ключевая цитата>",
    "why": "<почему это виральный момент>"
  }
}
```

### select_default.txt (RU)

```text
Ты — продюсер Reels/Shorts контента.

Из списка кандидатов выбери {count} лучших моментов для публикации.
Убедись, что моменты разнообразны и не пересекаются по времени.

Кандидаты:
{candidates_json}

Для каждого момента добавь:
- score (0-1): оценка виральности
- hook: интригующее начало для привлечения внимания
- caption: подпись для поста
- hashtags: релевантные хештеги

Ответь ТОЛЬКО валидным JSON:
{
  "moments": [
    {
      "start": <число>,
      "end": <число>,
      "title": "<заголовок>",
      "quote": "<цитата>",
      "why": "<причина>",
      "score": <0-1>,
      "hook": "<крючок>",
      "caption": "<подпись>",
      "hashtags": ["<тег1>", "<тег2>"]
    }
  ]
}
```

### chunk_default.txt (EN)

```text
You are an expert at creating viral social media content.

Analyze the transcript chunk below and find ONE most interesting moment with a duration from {r_min} to {r_max} seconds.

Virality criteria:
- Emotional statements
- Unexpected insights or confessions
- Humor or irony
- Controversial or provocative opinions
- Personal stories

Transcript (format: [start-end] text):
{transcript}

Reply with ONLY valid JSON:
{
  "moment": {
    "start": <number>,
    "end": <number>,
    "title": "<short title>",
    "quote": "<key quote>",
    "why": "<why it's viral>"
  }
}
```

### select_default.txt (EN)

```text
You are a Reels/Shorts content producer.

From the candidate list, pick {count} best moments for publishing.
Make sure moments are diverse and do not overlap in time.

Candidates:
{candidates_json}

For each moment add:
- score (0-1): virality score
- hook: intriguing opener
- caption: post caption
- hashtags: relevant hashtags

Reply with ONLY valid JSON:
{
  "moments": [
    {
      "start": <number>,
      "end": <number>,
      "title": "<title>",
      "quote": "<quote>",
      "why": "<why>",
      "score": <0-1>,
      "hook": "<hook>",
      "caption": "<caption>",
      "hashtags": ["<tag1>", "<tag2>"]
    }
  ]
}
```
