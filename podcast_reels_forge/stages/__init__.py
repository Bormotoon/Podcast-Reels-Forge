"""RU: Реализации стадий пайплайна, используемые root-скриптами.

Root-скрипты (`transcribe.py`, `analyze.py`, `video_processor.py`, `diarize.py`)
остаются тонкими entrypoint'ами: они перезапускаются в venv репозитория и затем
делегируют выполнение в эти модули.

EN: Pipeline stage implementations used by the root scripts.

The root scripts (`transcribe.py`, `analyze.py`, `video_processor.py`, `diarize.py`)
remain as thin entrypoints that re-exec into the repo venv and then delegate
into these modules.
"""
