    /* ===================================================================
       Podcast Reels Forge — Pipeline GUI
       Material Design 3.0 • Dark Theme • i18n (ru/en)
       =================================================================== */

    // =====================================================================
    //  INTERNATIONALIZATION (i18n)
    // =====================================================================
    const I18N = {
      ru: {
        // Nav
        nav_home: 'Главная', nav_transcribe: 'Транскрибация', nav_analyze: 'Анализ',
        nav_cut: 'Нарезка', nav_subtitles: 'Субтитры', nav_settings: 'Настройки', nav_logs: 'Логи',
        nav_dashboard: 'Главная', nav_subs: 'Субтитры',
        // Dashboard
        dash_title: 'Панель управления', dash_desc: 'Управление очередью пайплайна и мониторинг прогресса',
        dash_pipeline_status: 'Статус пайплайна',
        stage_transcribe: 'Транскрибация', stage_diarize: 'Диаризация', stage_analyze: 'Анализ LLM',
        stage_cut: 'Нарезка', stage_subtitles: 'Субтитры',
        stat_queue: 'В очереди', stat_reels: 'Рилсов создано', stat_duration: 'Общая длительность', stat_gpu: 'GPU VRAM',
        dash_add_queue: 'Добавить в очередь',
        dash_drop: 'Перетащите видео/аудио файлы сюда или нажмите для выбора',
        dash_formats: 'Поддерживаются: MP4, MKV, MOV, AVI, MP3',
        dash_run: 'Запустить пайплайн', dash_autotune: 'Автоподбор', dash_style_editor: 'Редактор стилей',
        dash_queue: 'Очередь обработки',
        dash_empty_title: 'Очередь пуста', dash_empty_desc: 'Загрузите видеофайлы для начала обработки',
        // Transcribe
        trans_title: 'Транскрибация', trans_desc: 'Настройка модели Whisper и параметров транскрибации',
        trans_model_device: 'Модель и устройство', trans_whisper_model: 'Модель Whisper',
        trans_device: 'Устройство', trans_compute_type: 'Тип вычислений', trans_language: 'Язык',
        trans_mode_quality: 'Режим и качество', trans_mode: 'Режим транскрибации',
        trans_fast: 'Быстрый (пакетный)', trans_quality: 'Качественный (последовательный)',
        trans_beam: 'Размер луча', trans_batch: 'Размер пакета (GPU)',
        trans_best_of: 'Лучший из', trans_qbeam: 'Лучший луч для качества',
        trans_antihall: 'Анти-галлюцинации и домен',
        trans_condition_text: 'Условие по предыдущему тексту',
        trans_condition_desc: 'Предотвращает петли галлюцинаций на тишине',
        trans_rep_penalty: 'Штраф за повтор', trans_no_repeat: 'Размер n-граммы без повтора',
        trans_patience: 'Терпение',
        trans_prompt: 'Начальный промпт (контекст домена)',
        trans_prompt_ph: 'Подсказка для лексики. Пример: Подкаст о технологиях и стартапах.',
        opt_auto_detect: 'Автоопределение',
        // Analyze
        an_title: 'Анализ', an_desc: 'LLM-поиск вирусных моментов и оценка',
        an_service: 'LLM-сервис', an_url: 'URL сервиса',
        an_autostart: 'Авто-запуск llama-server',
        an_autostart_desc: 'Запустить локальный сервер если не запущен',
        an_model_path: 'Путь к модели',
        an_roles: 'Роли моделей',
        an_scout: 'Разведчик (быстрый поиск)', an_cleanup: 'Очистка / Уточнение',
        an_judge: 'Судья / Метаданные',
        an_params: 'Параметры инференса', an_temp: 'Температура',
        an_timeout: 'Таймаут (сек)', an_chunk: 'Длительность чанка (сек)',
        an_max_chars: 'Макс. символов на чанк',
        an_watchdog: 'Watchdog и фоллбэк',
        an_wd_enabled: 'Watchdog включён', an_wd_desc: 'Обнаружение зависаний и перезапуск',
        an_wd_first: 'Таймаут первого токена', an_wd_stall: 'Таймаут зависания',
        an_wd_retries: 'Макс. попыток',
        an_prompts: 'Промпты и язык', an_prompt_lang: 'Язык промптов',
        an_variant: 'Вариант промпта', an_default: 'По умолчанию',
        // Cut
        cut_title: 'Нарезка и экспорт', cut_desc: 'Обработка видео, фильтры качества и форматы экспорта',
        cut_encoding: 'Кодирование',
        cut_nvenc: 'Использовать NVENC (GPU)',
        cut_nvenc_desc: 'Быстрое кодирование через NVIDIA GPU',
        cut_vbitrate: 'Видео битрейт', cut_abitrate: 'Аудио битрейт',
        cut_preset: 'Пресет (libx264 фоллбэк)', cut_nvenc_preset: 'NVENC пресет',
        cut_filters: 'Фильтры качества',
        cut_min_score: 'Мин. вирусная оценка', cut_min_dur: 'Мин. длительность (сек)',
        cut_max_dur: 'Макс. длительность (сек)', cut_face_ratio: 'Мин. доля лица',
        cut_crop: 'Кроп и детекция лица',
        cut_vertical: 'Вертикальный кроп',
        cut_vertical_desc: 'Обрезка горизонтального видео до 9:16',
        cut_smart: 'Умный кроп (трекинг лица)',
        cut_smart_desc: 'Следить за лицом спикера в кадре',
        cut_face_samples: 'Сэмплы лица', cut_face_min: 'Мин. размер лица (px)',
        cut_export: 'Форматы экспорта',
        cut_webm: 'Экспорт WebM', cut_webm_desc: 'Дополнительный формат WebM',
        cut_gif: 'Экспорт GIF', cut_gif_desc: 'Анимированный GIF-превью',
        cut_audio_only: 'Только аудио', cut_audio_only_desc: 'Экспорт MP3 аудиодорожки',
        cut_processing: 'Параметры обработки',
        cut_threads: 'Параллельные потоки', cut_padding: 'Паддинг рилсов (сек)',
        // Subtitles
        sub_title: 'Субтитры', sub_desc: 'Прошивка субтитров в рилсы с ASS-стилизацией',
        sub_settings: 'Настройки субтитров',
        sub_burn: 'Прошить субтитры', sub_burn_desc: 'Рендерить субтитры прямо в видео',
        sub_font: 'Шрифт', sub_ass: 'Файл ASS-стиля',
        sub_font_size: 'Размер шрифта (px)',
        sub_layout: 'Раскладка и анимация',
        sub_wrap: 'Перенос слов', sub_wrap_desc: 'Переносить длинные строки на несколько рядов',
        sub_max_lines: 'Макс. строк', sub_max_width: 'Макс. ширина (доля)',
        sub_valign: 'Вертикальное выравнивание',
        sub_top: 'Сверху', sub_center: 'По центру', sub_bottom: 'Снизу',
        sub_fade_in: 'Появление (сек)', sub_fade_out: 'Исчезновение (сек)',
        // Settings
        set_title: 'Настройки', set_desc: 'Общая конфигурация, пути и системные параметры',
        set_paths: 'Пути', set_input_dir: 'Директория ввода', set_output_dir: 'Директория вывода',
        set_cache_diar: 'Кэш и диаризация',
        set_cache: 'Кэш включён', set_cache_desc: 'Пропускать существующие файлы вывода',
        set_validate: 'Валидация JSON', set_validate_desc: 'Проверять целостность JSON-вывода',
        set_diar: 'Диаризация (ID спикеров)', set_diar_desc: 'Определять спикеров в аудио',
        set_diar_model: 'Модель диаризации',
        set_cli: 'Параметры CLI',
        set_quiet: 'Тихий режим', set_quiet_desc: 'Подавлять вывод не ошибок',
        set_verbose: 'Подробный режим', set_verbose_desc: 'Показывать подробные логи',
        set_skip: 'Пропускать завершённые', set_skip_desc: 'Не перезапускать завершённые стадии',
        set_preview: 'Предпросмотр config.yaml',
        set_export: 'Экспорт config.yaml', set_reset: 'Сбросить к умолчаниям',
        // Logs
        log_title: 'Логи пайплайна', log_desc: 'Вывод в реальном времени от стадий пайплайна',
        log_console: 'Консольный вывод', log_clear: 'Очистить', log_copy: 'Копировать',
        log_waiting: 'Ожидание запуска пайплайна...',
        // Chips
        chip_auto: 'Авто',
        // Bottom
        bottom_ready: 'Готов', bottom_running: 'Пайплайн работает...',
        bottom_complete: 'Пайплайн завершён', bottom_error: 'Произошла ошибка',
        // Log messages
        log_loaded: 'GUI пайплайна загружен',
        log_config_loaded: 'Конфигурация загружена из localStorage',
        log_queue_empty: 'Очередь пуста. Сначала добавьте видеофайлы.',
        log_no_files: 'Нет файлов в очереди. Сначала добавьте видеофайлы.',
        log_logs_cleared: 'Логи очищены',
        log_logs_copied: 'Логи скопированы в буфер обмена',
        log_config_exported: 'config.yaml экспортирован',
        log_config_reset: 'Конфигурация сброшена к умолчаниям',
        log_autotune_start: '[autotune] обнаружение конфигурации системы...',
        log_autotune_gpu: '[autotune] GPU:',
        log_autotune_nowebgpu: '[autotune] WebGPU адаптер недоступен',
        log_autotune_cpu: '[autotune] CPU ядра:',
        log_autotune_done: '[autotune] готово',
        log_forge_processing: '[forge] обработка:',
        log_stage_start: 'старт',
        log_stage_done: 'готово',
        log_stage_error: 'ошибка: имитация сбоя',
        log_forge_done: '[forge] готово:',
        log_forge_all_done: '[forge] все файлы обработаны',
        log_error_occurred: 'Произошла ошибка',
        log_pipeline_complete: 'Пайплайн завершён',
        // Analyze — service tuning & extras
        an_scout_par: 'Параллелизм разведчика', an_wd_log: 'Интервал логов (сек)',
        an_service_tuning: 'Сервис llama.cpp — VRAM и производительность',
        an_ngpu: 'Слои на GPU (n_gpu_layers)', an_ctx: 'Размер контекста (ctx_size)',
        an_startup: 'Таймаут старта (сек)', an_lbatch: 'Размер батча (batch_size)',
        an_ubatch: 'Микро-батч (ubatch_size)', an_lthreads: 'Потоки CPU (threads)',
        an_parallel: 'Параллельные слоты (parallel)', an_main_gpu: 'Основной GPU (main_gpu)',
        an_cache_k: 'KV-кэш K (cache_type_k)', an_cache_v: 'KV-кэш V (cache_type_v)',
        an_fallback: 'Фоллбэк-модели (через запятую)', an_fallback_ph: 'например: gemma4:12b, gemma4',
        an_extra_args: 'Доп. аргументы llama-server (через пробел)', an_extra_args_ph: 'например: --flash-attn --no-mmap',
        an_role_overrides: 'Переопределения по ролям',
        an_role_overrides_desc: 'Точечные настройки таймаута/температуры для каждой роли поверх общих параметров выше.',
        an_ro_scout_timeout: 'Разведчик: таймаут', an_ro_scout_temp: 'Разведчик: температура', an_ro_scout_chunk: 'Разведчик: чанк (сек)',
        an_ro_cleanup_timeout: 'Очистка: таймаут', an_ro_cleanup_temp: 'Очистка: температура',
        an_ro_judge_timeout: 'Судья: таймаут', an_ro_judge_temp: 'Судья: температура',
        // Cut — clips
        cut_clips: 'Типы клипов и количество',
        clips_stories_n: 'Stories — кол-во', clips_stories_d: 'Stories — макс. сек',
        clips_reels_n: 'Reels — кол-во', clips_reels_d: 'Reels — макс. сек',
        clips_long_n: 'Long reels — кол-во', clips_long_d: 'Long reels — макс. сек',
        clips_hl_n: 'Highlights — кол-во', clips_hl_m: 'Highlights — моментов',
        clips_reels_count: 'Reels всего (reels_count)', clips_reel_min: 'Reel мин. сек',
        clips_reel_max: 'Reel макс. сек',
        // Subtitles — extras
        sub_voffset: 'Верт. смещение (доля)', sub_word_x: 'Зазор слов X (px)', sub_word_y: 'Зазор слов Y (px)',
        sub_editor: 'Визуальный редактор стиля (.ass)', sub_editor_newtab: 'Открыть в новой вкладке',
        sub_editor_note: 'Все настройки субтитров — в одной панели слева от предпросмотра: параметры рендера, шрифт, цвет, контур и положение. Нажмите «Выбрать папку» и «Сохранить файл ASS» — пайплайн берёт стиль из этого файла.',
        sub_size_hint: 'Размер, цвет, контур и положение субтитров задаются ниже — в визуальном редакторе стиля (.ass).',
        // Settings — extra
        set_no_progress: 'Без прогресс-бара', set_no_progress_desc: 'Отключить UI прогресса (для логов/CI)',
        // FAB tooltip
        fab_run: 'Запустить пайплайн',
      },
      en: {
        nav_home: 'Home', nav_transcribe: 'Transcribe', nav_analyze: 'Analyze',
        nav_cut: 'Cut', nav_subtitles: 'Subtitles', nav_settings: 'Settings', nav_logs: 'Logs',
        nav_dashboard: 'Dashboard', nav_subs: 'Subtitles',
        dash_title: 'Dashboard', dash_desc: 'Manage your pipeline queue and monitor progress',
        dash_pipeline_status: 'Pipeline Status',
        stage_transcribe: 'Transcribe', stage_diarize: 'Diarize', stage_analyze: 'LLM Analyze',
        stage_cut: 'Cut & Export', stage_subtitles: 'Subtitles',
        stat_queue: 'In Queue', stat_reels: 'Reels Created', stat_duration: 'Total Duration', stat_gpu: 'GPU VRAM',
        dash_add_queue: 'Add to Queue',
        dash_drop: 'Drop video/audio files here or click to browse',
        dash_formats: 'Supports MP4, MKV, MOV, AVI, MP3',
        dash_run: 'Run Full Pipeline', dash_autotune: 'Autotune', dash_style_editor: 'Style Editor',
        dash_queue: 'Processing Queue',
        dash_empty_title: 'Queue is empty', dash_empty_desc: 'Upload video files to start processing',
        trans_title: 'Transcription', trans_desc: 'Configure Whisper model and transcription parameters',
        trans_model_device: 'Model & Device', trans_whisper_model: 'Whisper Model',
        trans_device: 'Device', trans_compute_type: 'Compute Type', trans_language: 'Language',
        trans_mode_quality: 'Mode & Quality', trans_mode: 'Transcription Mode',
        trans_fast: 'Fast (batched)', trans_quality: 'Quality (sequential)',
        trans_beam: 'Beam Size', trans_batch: 'Batch Size (GPU)',
        trans_best_of: 'Best Of', trans_qbeam: 'Quality Beam Size',
        trans_antihall: 'Anti-Hallucination & Domain',
        trans_condition_text: 'Condition on Previous Text',
        trans_condition_desc: 'Prevents hallucination loops on silence',
        trans_rep_penalty: 'Repetition Penalty', trans_no_repeat: 'No-Repeat N-gram Size',
        trans_patience: 'Patience',
        trans_prompt: 'Initial Prompt (domain context)',
        trans_prompt_ph: 'Optional domain hint to bias vocabulary. Example: Podcast about technology and startups.',
        opt_auto_detect: 'Auto-detect',
        an_title: 'Analysis', an_desc: 'LLM-based viral moment detection and scoring',
        an_service: 'LLM Service', an_url: 'Service URL',
        an_autostart: 'Auto-start llama-server',
        an_autostart_desc: 'Start local server automatically if not running',
        an_model_path: 'Model Path',
        an_roles: 'Model Roles',
        an_scout: 'Scout (fast scan)', an_cleanup: 'Cleanup / Refine',
        an_judge: 'Judge / Metadata',
        an_params: 'Inference Parameters', an_temp: 'Temperature',
        an_timeout: 'Timeout (sec)', an_chunk: 'Chunk Seconds',
        an_max_chars: 'Max Chars per Chunk',
        an_watchdog: 'Watchdog & Fallback',
        an_wd_enabled: 'Watchdog enabled', an_wd_desc: 'Detect stalling and restart inference',
        an_wd_first: 'First Token Timeout', an_wd_stall: 'Stall Timeout',
        an_wd_retries: 'Max Retries',
        an_prompts: 'Prompts & Language', an_prompt_lang: 'Prompt Language',
        an_variant: 'Prompt Variant', an_default: 'Default',
        cut_title: 'Cut & Export', cut_desc: 'Video processing, quality filters, and export formats',
        cut_encoding: 'Encoding',
        cut_nvenc: 'Use NVENC (GPU encoding)',
        cut_nvenc_desc: 'Faster encoding via NVIDIA GPU',
        cut_vbitrate: 'Video Bitrate', cut_abitrate: 'Audio Bitrate',
        cut_preset: 'Software Preset (libx264 fallback)', cut_nvenc_preset: 'NVENC Preset',
        cut_filters: 'Quality Filters',
        cut_min_score: 'Min Viral Score', cut_min_dur: 'Min Duration (sec)',
        cut_max_dur: 'Max Duration (sec)', cut_face_ratio: 'Min Face Ratio',
        cut_crop: 'Crop & Face Detection',
        cut_vertical: 'Vertical Crop',
        cut_vertical_desc: 'Crop horizontal video to 9:16',
        cut_smart: 'Smart Crop (Face Tracking)',
        cut_smart_desc: 'Follow speaker\'s face in the frame',
        cut_face_samples: 'Face Samples', cut_face_min: 'Face Min Size (px)',
        cut_export: 'Export Formats',
        cut_webm: 'Export WebM', cut_webm_desc: 'Additional WebM format output',
        cut_gif: 'Export GIF', cut_gif_desc: 'Animated GIF preview',
        cut_audio_only: 'Audio Only', cut_audio_only_desc: 'Export MP3 audio track',
        cut_processing: 'Processing Queue',
        cut_threads: 'Parallel Threads', cut_padding: 'Reel Padding (sec)',
        sub_title: 'Subtitles', sub_desc: 'Burn subtitles into reels with ASS styling',
        sub_settings: 'Subtitle Settings',
        sub_burn: 'Burn Subtitles', sub_burn_desc: 'Render subtitles directly into video',
        sub_font: 'Font', sub_ass: 'ASS Style File',
        sub_font_size: 'Font Size (px)',
        sub_layout: 'Layout & Animation',
        sub_wrap: 'Wrap Words', sub_wrap_desc: 'Wrap long lines across multiple rows',
        sub_max_lines: 'Max Lines', sub_max_width: 'Max Width Ratio',
        sub_valign: 'Vertical Align',
        sub_top: 'Top', sub_center: 'Center', sub_bottom: 'Bottom',
        sub_fade_in: 'Fade In (sec)', sub_fade_out: 'Fade Out (sec)',
        set_title: 'Settings', set_desc: 'General configuration, paths, and system options',
        set_paths: 'Paths', set_input_dir: 'Input Directory', set_output_dir: 'Output Directory',
        set_cache_diar: 'Cache & Diarization',
        set_cache: 'Cache Enabled', set_cache_desc: 'Skip existing output files',
        set_validate: 'Validate JSON', set_validate_desc: 'Verify output JSON integrity',
        set_diar: 'Diarization (Speaker ID)', set_diar_desc: 'Identify speakers in the audio',
        set_diar_model: 'Diarization Model',
        set_cli: 'CLI Options',
        set_quiet: 'Quiet Mode', set_quiet_desc: 'Suppress non-error output',
        set_verbose: 'Verbose Mode', set_verbose_desc: 'Show detailed logs',
        set_skip: 'Skip Existing', set_skip_desc: 'Don\'t reprocess completed stages',
        set_preview: 'Generated config.yaml Preview',
        set_export: 'Export config.yaml', set_reset: 'Reset to Defaults',
        log_title: 'Pipeline Logs', log_desc: 'Real-time output from pipeline stages',
        log_console: 'Console Output', log_clear: 'Clear', log_copy: 'Copy',
        log_waiting: 'Waiting for pipeline to start...',
        chip_auto: 'Auto',
        bottom_ready: 'Ready', bottom_running: 'Pipeline running...',
        bottom_complete: 'Pipeline complete', bottom_error: 'Error occurred',
        log_loaded: 'Pipeline GUI loaded',
        log_config_loaded: 'Config loaded from localStorage',
        log_queue_empty: 'Queue is empty. Add video files first.',
        log_no_files: 'No files in queue. Add video files first.',
        log_logs_cleared: 'Logs cleared',
        log_logs_copied: 'Logs copied to clipboard',
        log_config_exported: 'config.yaml exported',
        log_config_reset: 'Configuration reset to defaults',
        log_autotune_start: '[autotune] detecting system configuration...',
        log_autotune_gpu: '[autotune] GPU:',
        log_autotune_nowebgpu: '[autotune] No WebGPU adapter (running in worker?)',
        log_autotune_cpu: '[autotune] CPU cores:',
        log_autotune_done: '[autotune] done',
        log_forge_processing: '[forge] processing:',
        log_stage_start: 'start',
        log_stage_done: 'done',
        log_stage_error: 'error: simulated failure',
        log_forge_done: '[forge] done:',
        log_forge_all_done: '[forge] all files processed',
        log_error_occurred: 'Error occurred',
        log_pipeline_complete: 'Pipeline complete',
        an_scout_par: 'Scout Parallelism', an_wd_log: 'Log Interval (sec)',
        an_service_tuning: 'llama.cpp Service — VRAM & Performance',
        an_ngpu: 'GPU Layers (n_gpu_layers)', an_ctx: 'Context Size (ctx_size)',
        an_startup: 'Startup Timeout (sec)', an_lbatch: 'Batch Size (batch_size)',
        an_ubatch: 'Micro-batch (ubatch_size)', an_lthreads: 'CPU Threads (threads)',
        an_parallel: 'Parallel Slots (parallel)', an_main_gpu: 'Main GPU (main_gpu)',
        an_cache_k: 'KV Cache K (cache_type_k)', an_cache_v: 'KV Cache V (cache_type_v)',
        an_fallback: 'Fallback Models (comma-separated)', an_fallback_ph: 'e.g. gemma4:12b, gemma4',
        an_extra_args: 'Extra llama-server args (space-separated)', an_extra_args_ph: 'e.g. --flash-attn --no-mmap',
        an_role_overrides: 'Per-Role Overrides',
        an_role_overrides_desc: 'Fine-tune timeout/temperature per role on top of the global parameters above.',
        an_ro_scout_timeout: 'Scout: timeout', an_ro_scout_temp: 'Scout: temperature', an_ro_scout_chunk: 'Scout: chunk (sec)',
        an_ro_cleanup_timeout: 'Cleanup: timeout', an_ro_cleanup_temp: 'Cleanup: temperature',
        an_ro_judge_timeout: 'Judge: timeout', an_ro_judge_temp: 'Judge: temperature',
        cut_clips: 'Clip Types & Counts',
        clips_stories_n: 'Stories — count', clips_stories_d: 'Stories — max sec',
        clips_reels_n: 'Reels — count', clips_reels_d: 'Reels — max sec',
        clips_long_n: 'Long reels — count', clips_long_d: 'Long reels — max sec',
        clips_hl_n: 'Highlights — count', clips_hl_m: 'Highlights — moments',
        clips_reels_count: 'Total Reels (reels_count)', clips_reel_min: 'Reel min sec',
        clips_reel_max: 'Reel max sec',
        sub_voffset: 'Vertical Offset (ratio)', sub_word_x: 'Word Gap X (px)', sub_word_y: 'Word Gap Y (px)',
        sub_editor: 'Visual Style Editor (.ass)', sub_editor_newtab: 'Open in New Tab',
        sub_editor_note: 'Every subtitle setting lives in the single panel left of the preview: render parameters, font, color, outline and position. Use "Set Directory" and "Save ASS File" — the pipeline takes the style from that file.',
        sub_size_hint: 'Size, color, outline and position of subtitles are set below — in the visual style editor (.ass).',
        set_no_progress: 'No Progress Bar', set_no_progress_desc: 'Disable progress UI (for logs/CI)',
        fab_run: 'Run Pipeline',
      }
    };

    let currentLang = localStorage.getItem('forge.lang') || 'ru';

    function t(key) {
      return (I18N[currentLang] && I18N[currentLang][key]) || (I18N.ru[key]) || key;
    }

    function applyLang(lang) {
      currentLang = lang;
      localStorage.setItem('forge.lang', lang);
      document.documentElement.lang = lang;

      // Update all data-i18n elements
      document.querySelectorAll('[data-i18n]').forEach(el => {
        el.textContent = t(el.dataset.i18n);
      });

      // Update placeholders
      document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        el.placeholder = t(el.dataset.i18nPlaceholder);
      });

      // Update tooltips
      document.querySelectorAll('[data-tooltip-i18n]').forEach(el => {
        el.dataset.tooltip = t(el.dataset.tooltipI18n);
      });

      // Update lang toggle buttons
      document.querySelectorAll('.lang-toggle-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === lang);
      });

      // Update <title>
      document.title = lang === 'ru'
        ? 'Podcast Reels Forge — GUI пайплайна'
        : 'Podcast Reels Forge — Pipeline GUI';

      // Re-render dynamic elements
      renderQueue();
      updateBottomStatus();
    }

    // Language toggle click handlers
    document.addEventListener('click', e => {
      const btn = e.target.closest('.lang-toggle-btn');
      if (btn) applyLang(btn.dataset.lang);
    });

    // =====================================================================
    //  CONFIGURATION STATE
    // =====================================================================
    const STORAGE_KEY = 'forge.pipeline_gui.v2';

    const DEFAULTS = {
      transcribeModel: 'large-v3', transcribeDevice: 'auto', transcribeComputeType: 'auto',
      transcribeLanguage: 'auto', transcribeMode: 'fast', transcribeBeam: 5,
      transcribeBatch: 16, transcribeBestOf: 1, transcribeQBeam: 10,
      transcribeCondition: false, transcribeRepPenalty: 1.1, transcribeNoRepeat: 3,
      transcribePatience: 1.0, transcribePrompt: '',
      analyzeUrl: 'http://127.0.0.1:11440/completion', analyzeAutoStart: true,
      analyzeModelPath: '/opt/llamacpp/models/gemma4-26b.gguf',
      analyzeScout: 'gemma4:26b', analyzeCleanup: 'gemma4:26b', analyzeJudge: 'gemma4:26b',
      analyzeTemp: 0.2, analyzeTimeout: 600, analyzeChunk: 900, analyzeMaxChars: 7000,
      analyzeWatchdog: true, analyzeFirstToken: 90, analyzeStall: 120, analyzeRetries: 4,
      analyzeLogInterval: 10, analyzeScoutParallelism: 1, analyzeFallback: '',
      analyzeLang: 'auto', analyzeVariant: 'default',
      // llama.cpp service tuning (VRAM / performance)
      llamaNgpu: 0, llamaCtx: 8192, llamaBatch: 1024, llamaUbatch: 512,
      llamaThreads: 4, llamaParallel: 1, llamaMainGpu: 0, llamaStartupTimeout: 120,
      llamaCacheK: 'q8_0', llamaCacheV: 'q8_0', llamaExtraArgs: '',
      // Per-role overrides (llama_cpp.role_overrides.*)
      roScoutTimeout: 600, roScoutTemp: 0.35, roScoutChunk: 1200,
      roCleanupTimeout: 600, roCleanupTemp: 0.15,
      roJudgeTimeout: 600, roJudgeTemp: 0.05,
      cutNvenc: true, cutVBitrate: '8M', cutABitrate: '192k', cutPreset: 'fast',
      cutNvencCq: 21, cutNvencPreset: 'p5', cutMinScore: 7, cutMinDur: 15,
      cutMaxDur: 180, cutFaceRatio: 0.3, cutVertical: true, cutSmartCrop: true,
      cutFaceSamples: 9, cutFaceMinSize: 72, cutWebm: false, cutGif: false,
      cutAudio: false, cutThreads: 4, cutPadding: 5,
      // Clip selection / output mix (processing.clips + reels_*)
      procReelsCount: 3, procReelMinDur: 30, procReelMaxDur: 60,
      clipsStoriesCount: 2, clipsStoriesMaxDur: 15,
      clipsReelsCount: 3, clipsReelsMaxDur: 60,
      clipsLongCount: 1, clipsLongMaxDur: 180,
      clipsHlCount: 1, clipsHlMoments: 5,
      subsEnabled: true,
      subsFont: 'assets/fonts/bignoodletoooblique.ttf',
      subsAss: 'assets/subtitles/forge_subtitles.ass',
      subsWrap: true, subsMaxLines: 2, subsMaxWidth: 0.65,
      subsVOffset: 0.0, subsWordX: 6, subsWordY: 8,
      subsFadeIn: 0.18, subsFadeOut: 0.12,
      settingsInputDir: 'input', settingsOutputDir: 'output',
      settingsCache: true, settingsValidateJson: true, settingsDiarization: false,
      settingsDiarModel: 'pyannote/speaker-diarization',
      settingsQuiet: false, settingsVerbose: false, settingsSkipExisting: true,
      settingsNoProgress: false,
    };

    const state = { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') };
    let queue = [];
    let pipelineRunning = false;
    let currentStage = null;
    // Logs persist across page navigations within the session (mock pipeline run lives on the dashboard).
    let logLines = JSON.parse(sessionStorage.getItem('forge.logs') || '[]');

    // ---- Navigation ----
    // Nav items are real links now (separate pages); just mark the current page active.
    const CURRENT_PAGE = document.body.dataset.page || 'dashboard';
    document.querySelectorAll('.nav-rail-item').forEach(item => {
      item.classList.toggle('active', item.dataset.tab === CURRENT_PAGE);
    });

    // ---- Chip groups ----
    document.querySelectorAll('.chip[data-field]').forEach(chip => {
      chip.addEventListener('click', () => {
        const field = chip.dataset.field;
        document.querySelectorAll(`.chip[data-field="${field}"]`).forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        document.getElementById(field).value = chip.dataset.value;
        state[field.replace('cfg', '')] = chip.dataset.value;
        saveState();
      });
    });

    // ---- Sync all inputs to state ----
    function syncFromState() {
      Object.keys(DEFAULTS).forEach(key => {
        const el = document.getElementById('cfg' + key.charAt(0).toUpperCase() + key.slice(1));
        if (!el) return;
        if (el.type === 'checkbox') el.checked = state[key];
        else if (el.type === 'range') { el.value = state[key]; updateSliderValue(el); }
        else el.value = state[key];
        document.querySelectorAll(`.chip[data-field="cfg${key.charAt(0).toUpperCase() + key.slice(1)}"]`).forEach(c => {
          c.classList.toggle('active', c.dataset.value === String(state[key]));
        });
      });
      updateConfigPreview();
    }

    function updateSliderValue(el) {
      const valEl = document.getElementById('v-' + el.id);
      if (valEl) valEl.textContent = el.value;
    }

    document.querySelectorAll('input[type="range"]').forEach(el => {
      if (!el.id || !el.id.startsWith('cfg')) return;  // leave the embedded ASS editor's sliders alone
      el.addEventListener('input', () => {
        state[el.id.replace('cfg', '').replace(/^./, c => c.toLowerCase())] = parseFloat(el.value);
        updateSliderValue(el);
        saveState();
        updateConfigPreview();
      });
    });

    document.querySelectorAll('input[type="text"], input[type="number"], textarea, select').forEach(el => {
      if (el.id && el.id.startsWith('cfg')) {
        el.addEventListener('input', () => {
          state[el.id.replace('cfg', '').replace(/^./, c => c.toLowerCase())] = el.value;
          saveState();
          updateConfigPreview();
        });
      }
    });

    document.querySelectorAll('input[type="checkbox"]').forEach(el => {
      if (el.id && el.id.startsWith('cfg')) {
        el.addEventListener('change', () => {
          state[el.id.replace('cfg', '').replace(/^./, c => c.toLowerCase())] = el.checked;
          saveState();
          updateConfigPreview();
        });
      }
    });

    // The editor's font path (subtitles page) drives subtitles.font in the generated config.
    // Mirror it into state so the config exports correctly from ANY page, not just subtitles.
    const fontPathEl = document.getElementById('fontPath');
    if (fontPathEl) {
      const mirrorFont = () => { state.subsFont = fontPathEl.value.trim(); saveState(); updateConfigPreview(); };
      fontPathEl.addEventListener('input', mirrorFont);
    }

    function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }

    // ---- File drop zone (dashboard only) ----
    const fileDrop = document.getElementById('fileDrop');
    const fileInput = document.getElementById('fileInput');
    if (fileDrop && fileInput) {
      fileDrop.addEventListener('dragover', e => { e.preventDefault(); fileDrop.classList.add('dragover'); });
      fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('dragover'));
      fileDrop.addEventListener('drop', e => { e.preventDefault(); fileDrop.classList.remove('dragover'); addFilesToQueue(e.dataTransfer.files); });
      fileInput.addEventListener('change', () => { addFilesToQueue(fileInput.files); fileInput.value = ''; });
    }

    function addFilesToQueue(files) {
      for (const file of files) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (['mp4','mkv','mov','avi','mp3','wav','flac','ogg'].includes(ext)) {
          queue.push({ id: Date.now() + Math.random(), name: file.name, size: file.size, status: 'pending', file });
        }
      }
      renderQueue();
      updateStats();
    }

    function formatSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function renderQueue() {
      const list = document.getElementById('queueList');
      const empty = document.getElementById('queueEmpty');
      if (!list || !empty) return;  // queue UI lives on the dashboard only
      if (queue.length === 0) { empty.style.display = 'block'; list.querySelectorAll('.queue-item').forEach(el => el.remove()); return; }
      empty.style.display = 'none';
      list.querySelectorAll('.queue-item').forEach(el => el.remove());
      const statusLabels = { pending: t('stat_queue').toUpperCase(), running: t('bottom_running').toUpperCase().slice(0,8), done: 'DONE', error: 'ERROR' };
      queue.forEach(item => {
        const el = document.createElement('div');
        el.className = 'queue-item';
        el.innerHTML = `
          <div class="queue-item-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg></div>
          <div><div class="queue-item-name">${item.name}</div><div class="queue-item-meta">${formatSize(item.size)}</div></div>
          <span class="queue-item-status status-${item.status}">${statusLabels[item.status] || item.status.toUpperCase()}</span>
          <button class="btn btn-text" onclick="removeFromQueue('${item.id}')" style="min-width:auto; padding:0 8px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>`;
        list.appendChild(el);
      });
    }

    window.removeFromQueue = function(id) { queue = queue.filter(i => String(i.id) !== String(id)); renderQueue(); updateStats(); };
    function updateStats() { const el = document.getElementById('statQueue'); if (el) el.textContent = queue.length; }

    // ---- Pipeline Visual ----
    function setPipelineStage(stage, status) {
      const visual = document.getElementById('pipelineVisual');
      const overview = document.getElementById('pipelineOverview');
      if (!visual && !overview) return;  // dashboard pipeline visual / footer overview
      const stages = ['transcribe', 'diarize', 'analyze', 'cut', 'subtitles'];
      const idx = stages.indexOf(stage);
      if (visual) {
        if (status === 'active') {
          currentStage = stage;
          stages.forEach((s, i) => {
            const el = visual.querySelector(`[data-stage="${s}"]`);
            const conn = visual.querySelectorAll('.pipeline-connector')[i];
            if (!el) return;
            if (i < idx) { el.className = 'pipeline-stage done'; if (conn) conn.className = 'pipeline-connector done'; }
            else if (i === idx) { el.className = 'pipeline-stage active'; if (conn) conn.className = 'pipeline-connector active'; }
            else { el.className = 'pipeline-stage'; if (conn) conn.className = 'pipeline-connector'; }
          });
        } else if (status === 'done') {
          stages.forEach((s, i) => { if (i <= idx) { const el = visual.querySelector(`[data-stage="${s}"]`); if (el) el.className = 'pipeline-stage done'; const c = visual.querySelectorAll('.pipeline-connector')[i]; if (c) c.className = 'pipeline-connector done'; } });
        } else if (status === 'error') {
          const el = visual.querySelector(`[data-stage="${stage}"]`); if (el) el.className = 'pipeline-stage error';
        }
      }
      if (overview) overview.querySelectorAll('.pipeline-overview-dot').forEach(dot => {
        const si = stages.indexOf(dot.dataset.stage);
        dot.className = 'pipeline-overview-dot' + (si < idx ? ' done' : si === idx ? ' ' + status : '');
      });
    }

    // ---- Logging ----
    // logLines is persisted to sessionStorage so the Logs page shows runs started on the Dashboard.
    function persistLogs() { try { sessionStorage.setItem('forge.logs', JSON.stringify(logLines.slice(-500))); } catch (_) {} }
    function logLineEl(l) {
      const line = document.createElement('div');
      line.className = `log-line ${l.level}`;
      line.innerHTML = `<span class="ts">[${l.ts}]</span> ${l.msg}`;
      return line;
    }
    function addLog(msg, level = 'info') {
      const now = new Date().toLocaleTimeString('en-GB', { hour12: false });
      const l = { ts: now, msg, level };
      logLines.push(l);
      persistLogs();
      const panel = document.getElementById('logPanel');
      if (panel) { panel.appendChild(logLineEl(l)); panel.scrollTop = panel.scrollHeight; }
    }
    function renderPersistedLogs() {
      const panel = document.getElementById('logPanel');
      if (!panel) return;
      panel.innerHTML = '';
      logLines.forEach(l => panel.appendChild(logLineEl(l)));
      panel.scrollTop = panel.scrollHeight;
    }

    function updateBottomStatus() {
      const el = document.getElementById('bottomStatus');
      if (!el) return;
      if (pipelineRunning) el.textContent = t('bottom_running');
      else el.textContent = t('bottom_ready');
    }

    document.getElementById('btnClearLogs')?.addEventListener('click', () => { logLines = []; persistLogs(); const p = document.getElementById('logPanel'); if (p) p.innerHTML = ''; addLog(t('log_logs_cleared'), 'info'); });
    document.getElementById('btnCopyLogs')?.addEventListener('click', () => { navigator.clipboard.writeText(logLines.map(l => `[${l.ts}] ${l.msg}`).join('\n')).then(() => addLog(t('log_logs_copied'), 'info')); });

    // ---- Config Export ----
    function generateConfig() {
      const fb = String(state.analyzeFallback || '').split(',').map(s => s.trim()).filter(Boolean);
      const fbYaml = fb.length ? '\n' + fb.map(m => `    - "${m}"`).join('\n') : ' []';
      // extra_args is whitespace-separated on screen, emitted as a YAML list under service.
      const ea = String(state.llamaExtraArgs || '').split(/\s+/).map(s => s.trim()).filter(Boolean);
      const eaYaml = ea.length ? '\n' + ea.map(a => `      - "${a}"`).join('\n') : ' []';
      // Font path is owned by the style editor's "Путь к шрифту" field (subtitles page).
      // Prefer the live editor value, fall back to the mirrored state so export works on every page.
      const subsFont = (document.getElementById('fontPath')?.value || state.subsFont || 'assets/fonts/bignoodletoooblique.ttf').trim();
      return `# Podcast Reels Forge — Generated by Pipeline GUI
# ${new Date().toISOString()}

paths:
  input_dir: "${state.settingsInputDir}"
  output_dir: "${state.settingsOutputDir}"
cli:
  quiet: ${state.settingsQuiet}
  verbose: ${state.settingsVerbose}
cache:
  enabled: ${state.settingsCache}
  validate_json: ${state.settingsValidateJson}
transcription:
  model: "${state.transcribeModel}"
  device: "${state.transcribeDevice}"
  language: "${state.transcribeLanguage}"
  beam_size: ${state.transcribeBeam}
  compute_type: "${state.transcribeComputeType}"
  best_of: ${state.transcribeBestOf}
  patience: ${state.transcribePatience}
  batch_size: ${state.transcribeBatch}
  condition_on_previous_text: ${state.transcribeCondition}
  repetition_penalty: ${state.transcribeRepPenalty}
  no_repeat_ngram_size: ${state.transcribeNoRepeat}
  mode: "${state.transcribeMode}"
  quality_beam_size: ${state.transcribeQBeam}
  initial_prompt: "${state.transcribePrompt}"
llama_cpp:
  url: "${state.analyzeUrl}"
  service:
    auto_start: ${state.analyzeAutoStart}
    model_path: "${state.analyzeModelPath}"
    startup_timeout: ${state.llamaStartupTimeout}
    n_gpu_layers: ${state.llamaNgpu}
    ctx_size: ${state.llamaCtx}
    batch_size: ${state.llamaBatch}
    ubatch_size: ${state.llamaUbatch}
    threads: ${state.llamaThreads}
    main_gpu: ${state.llamaMainGpu}
    parallel: ${state.llamaParallel}
    cache_type_k: "${state.llamaCacheK}"
    cache_type_v: "${state.llamaCacheV}"
    extra_args:${eaYaml}
  scout_parallelism: ${state.analyzeScoutParallelism}
  roles:
    scout: "${state.analyzeScout}"
    cleanup_refine: "${state.analyzeCleanup}"
    judge_metadata: "${state.analyzeJudge}"
  timeout: ${state.analyzeTimeout}
  temperature: ${state.analyzeTemp}
  chunk_seconds: ${state.analyzeChunk}
  max_chars_chunk: ${state.analyzeMaxChars}
  watchdog:
    enabled: ${state.analyzeWatchdog}
    first_token_timeout: ${state.analyzeFirstToken}
    stall_timeout: ${state.analyzeStall}
    log_interval: ${state.analyzeLogInterval}
    max_retries: ${state.analyzeRetries}
  fallback_models:${fbYaml}
  role_overrides:
    scout:
      timeout: ${state.roScoutTimeout}
      chunk_seconds: ${state.roScoutChunk}
      temperature: ${state.roScoutTemp}
    cleanup_refine:
      timeout: ${state.roCleanupTimeout}
      temperature: ${state.roCleanupTemp}
    judge_metadata:
      timeout: ${state.roJudgeTimeout}
      temperature: ${state.roJudgeTemp}
  model_overrides: {}
prompts:
  language: "${state.analyzeLang}"
  variant: "${state.analyzeVariant}"
processing:
  quality_filters:
    min_score: ${state.cutMinScore}
    min_duration: ${state.cutMinDur}
    max_duration: ${state.cutMaxDur}
    face_min_ratio: ${state.cutFaceRatio}
  clips:
    stories:
      count: ${state.clipsStoriesCount}
      max_duration: ${state.clipsStoriesMaxDur}
    reels:
      count: ${state.clipsReelsCount}
      max_duration: ${state.clipsReelsMaxDur}
    long_reels:
      count: ${state.clipsLongCount}
      max_duration: ${state.clipsLongMaxDur}
    highlights:
      count: ${state.clipsHlCount}
      moments_count: ${state.clipsHlMoments}
  reels_count: ${state.procReelsCount}
  reel_min_duration: ${state.procReelMinDur}
  reel_max_duration: ${state.procReelMaxDur}
  reel_padding: ${state.cutPadding}
exports:
  webm: ${state.cutWebm}
  gif: ${state.cutGif}
  audio_only: ${state.cutAudio}
subtitles:
  enabled: ${state.subsEnabled}
  font: "${subsFont}"
  ass_style: "${state.subsAss}"
  font_size_px: 96  # запас, если нет .ass; реальный размер берёт редактор стиля
  wrap_words: ${state.subsWrap}
  max_lines: ${state.subsMaxLines}
  max_width_ratio: ${state.subsMaxWidth}
  vertical_align: "bottom"
  vertical_offset: ${state.subsVOffset}
  word_x_space: ${state.subsWordX}
  word_y_space: ${state.subsWordY}
  fade_in_duration: ${state.subsFadeIn}
  fade_out_duration: ${state.subsFadeOut}
video:
  threads: ${state.cutThreads}
  vertical_crop: ${state.cutVertical}
  smart_crop_face: ${state.cutSmartCrop}
  video_bitrate: "${state.cutVBitrate}"
  audio_bitrate: "${state.cutABitrate}"
  preset: "${state.cutPreset}"
  use_nvenc: ${state.cutNvenc}
  nvenc_cq: ${state.cutNvencCq}
  nvenc_preset: "${state.cutNvencPreset}"
  face_samples: ${state.cutFaceSamples}
  face_min_size: ${state.cutFaceMinSize}
diarization:
  enabled: ${state.settingsDiarization}
  model: "${state.settingsDiarModel}"
`;
    }

    function updateConfigPreview() { const el = document.getElementById('configPreview'); if (el) el.value = generateConfig(); }

    // ---- Config Export / Reset (settings page) ----
    document.getElementById('btnExportConfig')?.addEventListener('click', () => {
      const blob = new Blob([generateConfig()], { type: 'text/yaml' });
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'config.yaml'; a.click();
      addLog(t('log_config_exported'), 'success');
    });

    document.getElementById('btnResetConfig')?.addEventListener('click', () => {
      Object.assign(state, DEFAULTS); localStorage.removeItem(STORAGE_KEY); syncFromState();
      addLog(t('log_config_reset'), 'info');
    });

    // ---- Simulated Pipeline (dashboard) ----
    const btnRun = document.getElementById('btnRunPipeline');
    const fabRun = document.getElementById('fabRun');
    function setBottom(txt) { const el = document.getElementById('bottomStatus'); if (el) el.textContent = txt; }
    function setRunDisabled(v) { if (btnRun) btnRun.disabled = v; if (fabRun) fabRun.disabled = v; }

    async function runPipeline() {
      if (pipelineRunning) return;
      if (queue.length === 0) { addLog(t('log_no_files'), 'warn'); return; }
      pipelineRunning = true; setRunDisabled(true);
      const stages = ['transcribe', 'analyze', 'cut'];
      if (state.settingsDiarization) stages.splice(1, 0, 'diarize');
      if (state.subsEnabled) stages.push('subtitles');
      setBottom(t('bottom_running'));

      for (const item of queue) {
        if (item.status === 'done') continue;
        item.status = 'running'; renderQueue();
        addLog(`${t('log_forge_processing')} ${item.name}`, 'info');

        for (const stage of stages) {
          setPipelineStage(stage, 'active');
          addLog(`[${stage}] ${t('log_stage_start')}`, 'info');
          setBottom(`${stage}: ${item.name}`);
          await new Promise(r => setTimeout(r, 800 + Math.random() * 1200));
          if (Math.random() < 0.1) {
            setPipelineStage(stage, 'error');
            addLog(`[${stage}] ${t('log_stage_error')}`, 'error');
            item.status = 'error'; renderQueue();
            pipelineRunning = false; setRunDisabled(false);
            setBottom(t('bottom_error'));
            return;
          }
          setPipelineStage(stage, 'done');
          addLog(`[${stage}] ${t('log_stage_done')}`, 'success');
        }
        item.status = 'done'; renderQueue();
        addLog(`${t('log_forge_done')} ${item.name}`, 'success');
      }
      pipelineRunning = false; setRunDisabled(false);
      setBottom(t('bottom_complete'));
      const reelsEl = document.getElementById('statReels');
      if (reelsEl) reelsEl.textContent = queue.filter(i => i.status === 'done').length;
      addLog(t('log_forge_all_done'), 'success');
    }

    btnRun?.addEventListener('click', runPipeline);
    fabRun?.addEventListener('click', runPipeline);

    document.getElementById('btnAutotune')?.addEventListener('click', () => {
      addLog(t('log_autotune_start'), 'info');
      const gpuInfo = document.getElementById('gpuInfo');
      if (navigator.gpu) {
        navigator.gpu.requestAdapter().then(adapter => {
          if (gpuInfo) gpuInfo.textContent = `GPU: ${adapter.name || 'detected'}`;
          addLog(`${t('log_autotune_gpu')} ${adapter.name || 'detected'}`, 'success');
        }).catch(() => { if (gpuInfo) gpuInfo.textContent = 'GPU: --'; addLog(t('log_autotune_nowebgpu'), 'warn'); });
      }
      const cores = navigator.hardwareConcurrency || 4;
      const threads = Math.max(1, Math.min(8, Math.floor(cores / 2)));
      state.cutThreads = threads;
      const tEl = document.getElementById('cfgCutThreads');
      if (tEl) { tEl.value = threads; updateSliderValue(tEl); }
      addLog(`${t('log_autotune_cpu')} ${cores}, threads: ${threads}`, 'info');
      addLog(t('log_autotune_done'), 'success');
      saveState();
    });

    // ---- Style editor links ----
    // "Редактор стилей" on the dashboard → go to the subtitles page.
    document.getElementById('btnOpenStyleEditor')?.addEventListener('click', () => { location.href = 'subtitles.html'; });
    // "Открыть в новой вкладке" on the subtitles page → open the standalone editor.
    document.getElementById('btnEditorNewTab')?.addEventListener('click', () => window.open('../assets/subtitles/style-editor.html', '_blank'));
    // The embedded phone preview needs a resize tick once the subtitles page is laid out.
    if (CURRENT_PAGE === 'subtitles') {
      window.addEventListener('load', () => setTimeout(() => window.dispatchEvent(new Event('resize')), 120));
    }

    // ---- Init ----
    applyLang(currentLang);
    syncFromState();
    updateStats();
    renderPersistedLogs();
    if (CURRENT_PAGE === 'dashboard') {
      addLog(t('log_loaded'), 'info');
      addLog(`${t('log_config_loaded')} (${Object.keys(state).length})`, 'info');
    }
