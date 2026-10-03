  (function(){

    const STORAGE_KEY = "forge.assStyleStudio.m3_readable";
    
    const PLATFORMS = {
        ig: { 
            color: '#E1306C', 
            points: '35,220 1045,220 1045,800 940,800 940,1500 35,1500', 
            desc: '<strong>IG Reels:</strong> Безопасная зона. Отступы: Верх 220px, Низ 420px, Справа 140px.',
            ui: `
                <text x="40" y="120" font-size="55" class="svg-t">Reels</text>
                <g transform="translate(970, 70) scale(1.8)" class="svg-i" stroke="white" stroke-width="2" fill="none">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle>
                </g>
                <g transform="translate(970, 850) scale(2)" class="svg-i" fill="none" stroke="white" stroke-width="2">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                </g>
                <text x="995" y="930" font-size="24" class="svg-t" text-anchor="middle">345K</text>
                <g transform="translate(970, 980) scale(2)" class="svg-i" fill="none" stroke="white" stroke-width="2">
                    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
                </g>
                <text x="995" y="1060" font-size="24" class="svg-t" text-anchor="middle">1,234</text>
                <g transform="translate(970, 1110) scale(2)" class="svg-i" fill="none" stroke="white" stroke-width="2">
                    <line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </g>
                <text x="995" y="1190" font-size="24" class="svg-t" text-anchor="middle">12K</text>
                <g transform="translate(970, 1260) scale(2)" class="svg-i" fill="white">
                    <circle cx="12" cy="12" r="2"></circle><circle cx="12" cy="5" r="2"></circle><circle cx="12" cy="19" r="2"></circle>
                </g>
                <rect x="955" y="1360" width="80" height="80" rx="15" fill="#333" stroke="white" stroke-width="4" class="svg-i" />
                <circle cx="80" cy="1580" r="40" fill="#555" stroke="rgba(255,255,255,0.4)" stroke-width="2" class="svg-i" />
                <text x="140" y="1595" font-size="38" class="svg-t">@username</text>
                <rect x="375" y="1555" width="180" height="50" rx="15" stroke="white" stroke-width="2" fill="rgba(0,0,0,0.3)"/>
                <text x="465" y="1590" font-size="26" class="svg-t" text-anchor="middle">Подписаться</text>
                <text x="40" y="1670" font-size="36" class="svg-t">Ваша классная подпись будет здесь. Она переносится...</text>
                <g transform="translate(40, 1700) scale(1.4)" fill="white" class="svg-i">
                    <path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle>
                </g>
                <text x="95" y="1725" font-size="30" class="svg-t">Оригинальный звук — Исполнитель</text>
            `
        },
        tiktok: { 
            color: '#00E6E6', 
            points: '44,150 1036,150 1036,630 920,630 920,1440 44,1440', 
            desc: '<strong>TikTok:</strong> Безопасная зона. Отступы: Верх 150px, Низ 480px, Справа 160px.',
            ui: `
                <text x="450" y="100" font-size="42" class="svg-t" opacity="0.6">Подписки</text>
                <text x="650" y="100" font-size="42" class="svg-t">Для вас</text>
                <rect x="670" y="120" width="55" height="8" fill="white" rx="4" class="svg-i" />
                <g transform="translate(970, 50) scale(1.8)" class="svg-i" fill="none" stroke="white" stroke-width="2">
                    <circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </g>
                <g transform="translate(965, 650)">
                    <circle cx="25" cy="25" r="45" fill="white" class="svg-i"/>
                    <circle cx="25" cy="25" r="40" fill="#333" />
                    <circle cx="25" cy="65" r="18" fill="#FE2C55" class="svg-i"/>
                    <text x="25" y="76" font-size="32" class="svg-t" text-anchor="middle">+</text>
                </g>
                <g transform="translate(945, 800) scale(2.2)" class="svg-i" fill="white">
                    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"></path>
                </g>
                <text x="990" y="880" font-size="24" class="svg-t" text-anchor="middle">1.2M</text>
                <g transform="translate(945, 930) scale(2.2)" class="svg-i" fill="white">
                    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
                </g>
                <text x="990" y="1010" font-size="24" class="svg-t" text-anchor="middle">4564</text>
                <g transform="translate(945, 1060) scale(2.2)" class="svg-i" fill="white">
                    <path d="M17 3H7c-1.1 0-1.99.9-1.99 2L5 21l7-3 7 3V5c0-1.1-.9-2-2-2z"></path>
                </g>
                <text x="990" y="1140" font-size="24" class="svg-t" text-anchor="middle">123K</text>
                <g transform="translate(945, 1190) scale(2.2)" class="svg-i" fill="white">
                    <path d="M15 5l-1.41 1.41L18.17 11H2v2h16.17l-4.59 4.59L15 19l7-7-7-7z"></path>
                </g>
                <text x="990" y="1270" font-size="24" class="svg-t" text-anchor="middle">78K</text>
                <circle cx="990" cy="1380" r="45" fill="#222" class="svg-i"/>
                <circle cx="990" cy="1380" r="28" fill="#333" />
                <text x="44" y="1520" font-size="42" class="svg-t">@username</text>
                <text x="44" y="1575" font-size="38" class="svg-t">Текст подписи TikTok прямо здесь #viral</text>
                <g transform="translate(44, 1610) scale(1.5)" fill="white" class="svg-i">
                    <path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle>
                </g>
                <text x="95" y="1638" font-size="32" class="svg-t">Оригинальный звук — Автор</text>
            `
        },
        yt: { 
            color: '#FF0000', 
            points: '40,200 1040,200 1040,700 920,700 920,1550 40,1550', 
            desc: '<strong>YouTube Shorts:</strong> Безопасная зона. Отступы: Верх 200px, Низ 370px, Справа 160px.',
            ui: `
                <g transform="translate(860, 60) scale(1.8)" class="svg-i" fill="none" stroke="white" stroke-width="2">
                    <circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </g>
                <g transform="translate(970, 60) scale(1.8)" class="svg-i" fill="white">
                    <circle cx="12" cy="12" r="2"></circle><circle cx="12" cy="5" r="2"></circle><circle cx="12" cy="19" r="2"></circle>
                </g>
                <g transform="translate(945, 720) scale(2.2)" class="svg-i" fill="white">
                    <path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z"></path>
                </g>
                <text x="995" y="805" font-size="22" class="svg-t" text-anchor="middle">320K</text>
                <g transform="translate(945, 850) scale(2.2)" class="svg-i" fill="white">
                    <path d="M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.59-6.59c.36-.36.58-.86.58-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z"></path>
                </g>
                <text x="995" y="935" font-size="22" class="svg-t" text-anchor="middle">Не нравится</text>
                <g transform="translate(945, 980) scale(2.2)" class="svg-i" fill="white">
                    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
                </g>
                <text x="995" y="1065" font-size="22" class="svg-t" text-anchor="middle">1,234</text>
                <g transform="translate(945, 1110) scale(2.2)" class="svg-i" fill="white">
                    <path d="M15 5l-1.41 1.41L18.17 11H2v2h16.17l-4.59 4.59L15 19l7-7-7-7z"></path>
                </g>
                <text x="995" y="1195" font-size="22" class="svg-t" text-anchor="middle">Поделиться</text>
                <g transform="translate(945, 1240) scale(2.2) rotate(90 12 12)" class="svg-i" fill="white">
                    <path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm-8 8c0-1.01.25-1.97.7-2.8L3.24 7.74C2.46 8.97 2 10.43 2 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3c-3.31 0-6-2.69-6-6z"></path>
                </g>
                <text x="995" y="1325" font-size="22" class="svg-t" text-anchor="middle">Ремикс</text>
                <rect x="950" y="1380" width="90" height="90" rx="15" fill="#333" stroke="white" stroke-width="4" class="svg-i"/>
                <circle cx="85" cy="1590" r="45" fill="#555" stroke="rgba(255,255,255,0.4)" stroke-width="2" class="svg-i" />
                <text x="145" y="1605" font-size="40" class="svg-t">@НазваниеКанала</text>
                <rect x="470" y="1565" width="260" height="60" rx="30" fill="white" class="svg-i"/>
                <text x="600" y="1605" font-size="30" fill="black" font-weight="bold" font-family="'Roboto', sans-serif" text-anchor="middle">Подписаться</text>
                <text x="40" y="1690" font-size="40" class="svg-t">Это цепляющий заголовок YouTube Shorts...</text>
            `
        }
    };

    // Built-in looks shared with the burner (generated from
    // podcast_reels_forge/utils/subtitle_presets.py into subtitle-presets.js).
    const SHARED = window.FORGE_SUBTITLE_PRESETS || { lookDefaults: {}, presets: {} };

    const DEFAULTS = {
      fontPath: "assets/fonts/bignoodletoooblique.ttf",
      fontSizePx: 96, spacingPx: 0, bold: true, italic: false, underline: false, strikeout: false,
      // \kf заливает слово от вторичного цвета к основному: белое — ещё не
      // произнесено, янтарное — уже прозвучало. Толстый контур вместо тени —
      // именно он держит читаемость поверх любого видео.
      primaryColor: "#FFD60A", primaryOp: 1.0, secondaryColor: "#FFFFFF", secondaryOp: 1.0,
      borderStyle: 1, outlineColor: "#000000", outlineOp: 1.0, outline: 8,
      backColor: "#000000", backOp: 0.5, shadow: 0,
      // Поля 140px обходят правую панель кнопок Reels/TikTok/Shorts,
      // MarginV 470 поднимает текст над подписью и строкой со звуком.
      alignment: 2, marginV: 470, marginL: 140, marginR: 140,
      scaleX: 100, scaleY: 100, angle: 0,
      // Стиль «Highlight» — активное слово (subtitles.highlight).
      hlEnabled: false, hlColor: "#FFD60A", hlOp: 1.0, hlBorderStyle: 1,
      hlOutlineColor: "#000000", hlOutlineOp: 1.0, hlOutline: 8,
      hlBackColor: "#000000", hlBackOp: 0.5, hlShadow: 0, hlScale: 100,
      ...SHARED.lookDefaults,
      sampleText: "Разбираемся, почему этот выпуск вызывает столько споров у зрителей.", autoAnimate: true,
      platform: "ig", showUI: true, showMask: true, showOutline: true, maskOpacity: 60,
      activePreset: ""
    };

    // Render settings (config.yaml → subtitles.*) that a full preset sets.
    // They live in the "render parameters" block (cfgSubs* ids, owned by app.js).
    const RENDER_FIELDS = {
      highlight:          { id: 'cfgSubsHighlight', def: 'none' },
      text_case:          { id: 'cfgSubsCase', def: 'none' },
      strip_punctuation:  { id: 'cfgSubsPunct', def: 'keep' },
      line_balance:       { id: 'cfgSubsBalance', def: 'bottom_heavy' },
      max_words_per_cue:  { id: 'cfgSubsMaxWords', def: 0 },
      max_lines:          { id: 'cfgSubsMaxLines', def: 2 },
      blur:               { id: 'cfgSubsBlur', def: 0 },
      fade_in_duration:   { id: 'cfgSubsFadeIn', def: 0.12 },
      fade_out_duration:  { id: 'cfgSubsFadeOut', def: 0.08 },
      min_duration_s:     { id: 'cfgSubsMinDur', def: 1.5 },
    };

    function renderValue(key) {
      const f = RENDER_FIELDS[key];
      const el = f && document.getElementById(f.id);
      if (!el) return f ? f.def : undefined;
      if (el.type === 'checkbox') return el.checked;
      if (el.type === 'range' || el.type === 'number') return parseFloat(el.value);
      return el.value;
    }
    function cfgValue(id, def) {
      const el = document.getElementById(id);
      if (!el) return def;
      if (el.type === 'range' || el.type === 'number') return parseFloat(el.value);
      return el.value;
    }

    function setRenderField(key, value) {
      const f = RENDER_FIELDS[key];
      const el = f && document.getElementById(f.id);
      if (!el) return;
      if (el.type === 'checkbox') el.checked = !!value; else el.value = value;
      // Let app.js store it and refresh the config preview.
      el.dispatchEvent(new Event(el.type === 'checkbox' ? 'change' : 'input', { bubbles: true }));
    }

    // One click: the whole look (Default + Highlight styles) and the render
    // settings that go with it, exactly as subtitles.preset does in the burner.
    window.applyFullPreset = function(name) {
      const preset = SHARED.presets[name];
      if (!preset) return;
      Object.assign(state, DEFAULTS_LOOK(), preset.look || {});
      state.activePreset = name;
      const render = preset.render || {};
      Object.keys(RENDER_FIELDS).forEach(key => {
        setRenderField(key, key in render ? render[key] : RENDER_FIELDS[key].def);
      });
      const fontEl = document.getElementById('fontPath');
      if (fontEl) { fontEl.value = state.fontPath; fontEl.dispatchEvent(new Event('input', { bubbles: true })); }
      sync();
    };

    function DEFAULTS_LOOK() {
      const look = {};
      Object.keys(SHARED.lookDefaults).forEach(k => { look[k] = SHARED.lookDefaults[k]; });
      return look;
    }

    function buildPresetGrid() {
      const grid = document.getElementById('fullPresetGrid');
      if (!grid) return;
      const lang = (document.documentElement.lang || 'ru').startsWith('en') ? 'en' : 'ru';
      grid.innerHTML = '';
      Object.entries(SHARED.presets).forEach(([name, preset]) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'preset-btn full-preset-btn ripplable';
        btn.dataset.preset = name;
        btn.textContent = (preset.title && preset.title[lang]) || name;
        btn.title = (preset.description && preset.description[lang]) || '';
        btn.addEventListener('click', () => window.applyFullPreset(name));
        grid.appendChild(btn);
      });
    }

    // Пресеты, вдохновлённые самыми вирусными форматами роликов
    const PRESETS = {
      typo: {
        // Alex Hormozi: огромный жирный текст, слегка вытянутый по вертикали
        hormozi: { fontSizePx: 72, spacingPx: -1, bold: true, italic: false, scaleX: 100, scaleY: 108, angle: 0 },
        // MrBeast: максимально крупный, энергичный
        mrbeast: { fontSizePx: 84, spacingPx: 0, bold: true, italic: false, scaleX: 105, scaleY: 100, angle: 0 },
        // TikTok auto-caption / караоке
        tiktok: { fontSizePx: 60, spacingPx: 1, bold: true, italic: false, scaleX: 100, scaleY: 100, angle: 0 },
        // Минималистичный «Apple-style» тонкий шрифт
        minimal: { fontSizePx: 44, spacingPx: 3, bold: false, italic: false, scaleX: 100, scaleY: 100, angle: 0 },
        // Динамичный курсив с наклоном
        impact: { fontSizePx: 58, spacingPx: 0, bold: true, italic: true, scaleX: 100, scaleY: 100, angle: 4 }
      },
      colors: {
        hormozi: { primaryColor: "#FFD60A", secondaryColor: "#FFFFFF", primaryOp: 1.0, secondaryOp: 1.0 },
        green: { primaryColor: "#39FF14", secondaryColor: "#FFFFFF", primaryOp: 1.0, secondaryOp: 1.0 },
        tiktok: { primaryColor: "#00F2EA", secondaryColor: "#FFFFFF", primaryOp: 1.0, secondaryOp: 1.0 },
        red: { primaryColor: "#FF2D2D", secondaryColor: "#FFFFFF", primaryOp: 1.0, secondaryOp: 1.0 },
        white: { primaryColor: "#FFFFFF", secondaryColor: "#FFFFFF", primaryOp: 1.0, secondaryOp: 1.0 }
      },
      borders: {
        // Hormozi: толстый чёрный контур без тени
        hormozi: { borderStyle: 1, outline: 9, outlineColor: "#000000", outlineOp: 1.0, shadow: 0, backColor: "#000000", backOp: 0.8 },
        // Мягкая drop-shadow
        shadow: { borderStyle: 1, outline: 3, outlineColor: "#000000", outlineOp: 1.0, shadow: 7, backColor: "#000000", backOp: 0.75 },
        // Непрозрачная плашка (как авто-субтитры TikTok/YouTube)
        // BorderStyle 3 paints the box with OutlineColour (libass); Outline is its padding.
        box: { borderStyle: 3, outline: 12, outlineColor: "#000000", outlineOp: 0.9, backColor: "#000000", backOp: 0.9, shadow: 0 },
        // Плашка + белый контур
        // BorderStyle 4: BackColour box around the cue, white outline on the letters.
        box_outline: { borderStyle: 4, outline: 3, outlineColor: "#FFFFFF", outlineOp: 1.0, backColor: "#000000", backOp: 0.85, shadow: 0 },
        // Неоновое свечение
        neon: { borderStyle: 1, outline: 4, outlineColor: "#000000", outlineOp: 1.0, shadow: 10, backColor: "#00F2EA", backOp: 0.9 },
        // Одна плашка на всю реплику (BorderStyle 4, libass ≥ 0.17): контур = отступ
        box_cue: { borderStyle: 4, outline: 18, outlineColor: "#000000", outlineOp: 0.0, backColor: "#000000", backOp: 0.75, shadow: 0 },
        // «Наклейка»: очень толстый контур цвета плашки читается как скруглённая подложка
        sticker: { borderStyle: 1, outline: 20, outlineColor: "#FFFFFF", outlineOp: 1.0, shadow: 0, primaryColor: "#111111", secondaryColor: "#111111" },
        // Мягкая тень без контура
        soft: { borderStyle: 1, outline: 0, outlineOp: 1.0, shadow: 4, backColor: "#000000", backOp: 0.6 }
      },
      geometry: {
        reels: { alignment: 2, marginV: 470 },
        shorts: { alignment: 2, marginV: 360 },
        center: { alignment: 5, marginV: 0 },
        top: { alignment: 8, marginV: 250 },
        tiktok: { alignment: 2, marginV: 560 },
        // Нижняя треть кадра — между лицом и интерфейсом
        lower_third: { alignment: 2, marginV: 640 },
        // 2/3 снизу (как в MoneyPrinterTurbo): над головами, под верхней панелью
        upper_third: { alignment: 8, marginV: 560 },
        // Чуть ниже центра — не закрывает глаза говорящего
        below_center: { alignment: 8, marginV: 1080 }
      }
    };

    const state = { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") };
    let projHandle = null, activeIdx = 0, cueIdx = 0, currentCueLength = 0, objectUrl = null;

    // --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ АНИМАЦИИ ---
    let currentScale = 0.35;
    let rotX = 0;
    let rotY = 0;

    // ЕДИНАЯ функция применения трансформаций (убирает конфликты!)
    function applyTransforms() {
        const frame = document.getElementById('deviceFrame');
        if(!frame) return;
        frame.style.transform = `scale(${currentScale}) rotateX(${rotX}deg) rotateY(${rotY}deg)`;
    }

    // --- PARALLAX 3D HOVER EFFECT ON PHONE ---
    const workspace = document.getElementById('workspaceBody');
    workspace.addEventListener('mousemove', (e) => {
      const rect = workspace.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      
      // Вычисляем угол наклона (макс 6 градусов)
      rotX = ((y - centerY) / centerY) * -6; 
      rotY = ((x - centerX) / centerX) * 6;
      
      applyTransforms();
    });
    
    workspace.addEventListener('mouseleave', () => {
      rotX = 0; rotY = 0;
      applyTransforms();
    });

    // ТОЧНОЕ МАТЕМАТИЧЕСКОЕ МАСШТАБИРОВАНИЕ (Без багов)
    function resizePreview() {
      const body = document.getElementById('workspaceBody');
      const scaler = document.getElementById('deviceScaler');
      
      // Жестко зашитые физические размеры оболочки:
      // Экран 1080x1920 + Padding(34*2) + Borders(6*2)
      const frameW = 1160; 
      const frameH = 2000; 
      
      const rect = body.getBoundingClientRect();
      const maxW = rect.width - 40;
      const maxH = rect.height - 40;
      
      let scale = Math.min(maxW / frameW, maxH / frameH);
      if (scale <= 0 || isNaN(scale)) scale = 0.35;
      
      // Передаем размеры в резервный контейнер для Flex-центровки
      scaler.style.width = `${frameW * scale}px`;
      scaler.style.height = `${frameH * scale}px`;
      
      // Сохраняем глобальный масштаб и применяем
      currentScale = scale;
      applyTransforms();
    }

    window.applyPreset = function(category, name) {
      if(PRESETS[category] && PRESETS[category][name]) {
        Object.assign(state, PRESETS[category][name]);
        sync();
      }
    };

    function toASSColor(hex, opacity) {
      hex = hex.replace('#', '');
      if (hex.length === 3) hex = hex.split('').map(c => c+c).join('');
      let r = hex.substring(0,2).toUpperCase();
      let g = hex.substring(2,4).toUpperCase();
      let b = hex.substring(4,6).toUpperCase();
      let a = Math.round((1 - opacity) * 255).toString(16).padStart(2, '0').toUpperCase();
      return `&H${a}${b}${g}${r}`;
    }

    function toRGBA(hex, op) {
      hex = hex.replace('#', '');
      if (hex.length === 3) hex = hex.split('').map(c => c+c).join('');
      return `rgba(${parseInt(hex.substring(0,2),16)}, ${parseInt(hex.substring(2,4),16)}, ${parseInt(hex.substring(4,6),16)}, ${op})`;
    }

    // RU: Этот же разметочный блок открывается из gui/ (на уровень выше корня
    //     репозитория) и из assets/subtitles/ (на два). Жёсткий "../../" ломал
    //     шрифт в GUI: файл не находился и предпросмотр молча падал на sans-serif.
    //     @font-face перебирает src по списку, пока какой-то не загрузится.
    // EN: This markup is served both from gui/ (one level below the repo root)
    //     and assets/subtitles/ (two). A hardcoded "../../" broke the font in the
    //     GUI: it 404'd and the preview silently fell back to sans-serif.
    //     @font-face walks the src list until one of them loads.
    function repoRootPrefix() {
      const dir = new URL('.', document.baseURI).pathname;
      if (/\/gui\/$/.test(dir)) return '../';
      if (/\/assets\/subtitles\/$/.test(dir)) return '../../';
      return '';
    }

    function fontFaceSrc(rawPath) {
      const path = String(rawPath || '').trim();
      if (!path) return '';
      const esc = path.replace(/"/g, '\\"');
      if (/^(https?:|data:|file:|\/)/i.test(path)) return `url("${esc}")`;
      // Known depth first (so the normal case costs no 404), the rest as a
      // self-healing fallback if these files ever get moved.
      const prefixes = [repoRootPrefix(), '../', '../../', ''];
      const seen = new Set();
      return prefixes
        .filter(p => !seen.has(p) && seen.add(p))
        .map(p => `url("${p}${esc}")`)
        .join(', ');
    }

    let lastFontSrc = null;
    function applyFontFace() {
      const src = fontFaceSrc(state.fontPath);
      // Rewriting the rule on every repaint restarts the font fetch and makes
      // the preview flicker, so only touch it when the path actually changed.
      if (src === lastFontSrc) return;
      lastFontSrc = src;
      const styleEl = document.getElementById('dynamic-font-face');
      if (styleEl) {
        styleEl.textContent = src
          ? `@font-face { font-family: 'AssPreview'; src: ${src}; font-display: swap; }`
          : '';
      }
    }

    // ---- Text transforms & line layout: a port of utils/subtitle_layout.py ----
    // The preview breaks lines the way the burner does, so what you see in the
    // phone is what lands in the reel.
    const NO_LINE_END = new Set(("в на по из за к у о об от до со ко а и но ни да нет не то ли бы же вот " +
      "ну или что как где когда чтобы пока тоже уже ещё еще просто только ведь если либо однако потом " +
      "тогда сейчас потому раз хотя чтоб будто даже вообще именно конечно пожалуй пожалуйста сразу " +
      "типа кроме после перед между через около с без для при про над под из-за из-под " +
      "the a an of to in on at for and or but with from by").split(' '));
    const SENTENCE_END = /[.!?…]+[»"')\]]*$/;
    const CLAUSE_END = /[,;:—–]+[»"')\]]*$/;
    const bareWord = w => w.replace(/^[.,!?…;:«»"'()\[\]—–-]+|[.,!?…;:«»"'()\[\]—–-]+$/g, '').toLowerCase();

    function applyCase(word, mode) {
      if (mode === 'upper') return word.toUpperCase();
      if (mode === 'lower') return word.toLowerCase();
      if (mode === 'title') return word.split('-').map(p => p.charAt(0).toUpperCase() + p.slice(1).toLowerCase()).join('-');
      return word;
    }
    function stripPunct(word, mode) {
      if (mode === 'periods') {
        if (word.endsWith('...') || word.endsWith('…')) return word;
        return word.replace(/(?<![.…])[.,;:]+(?=[»"')\]]*$)/, '');
      }
      if (mode === 'all') return word.replace(/[^\p{L}\p{N}\s'’-]|(?<![\p{L}\p{N}])[-'’]|[-'’](?![\p{L}\p{N}])/gu, '');
      return word;
    }

    const measureCanvas = document.createElement('canvas').getContext('2d');
    // libass sizes a face so ascender+descender span Fontsize; CSS sizes the em.
    // Measure the loaded font's span once per family to convert between them.
    let assScaleCache = { key: '', value: 1 };
    function assFontScale() {
      // Shipped fonts: the exact OS/2 win metrics libass uses (browsers only
      // expose hhea/typo metrics, which differ e.g. for Montserrat).
      const known = (SHARED.fontMetrics || {})[String(state.fontPath).replace(/^\.\//, '')];
      if (known) return 1 / known;
      const key = `${state.fontPath}|${document.fonts ? document.fonts.status : ''}`;
      if (assScaleCache.key === key) return assScaleCache.value;
      measureCanvas.font = `100px AssPreview, sans-serif`;
      const m = measureCanvas.measureText('Hg');
      const span = (m.fontBoundingBoxAscent || 0) + (m.fontBoundingBoxDescent || 0);
      const value = span > 0 ? 100 / span : 1;
      assScaleCache = { key, value };
      return value;
    }
    function cssFontPx() { return state.fontSizePx * assFontScale(); }
    function textWidth(text) {
      measureCanvas.font = `${state.bold ? 'bold ' : ''}${state.italic ? 'italic ' : ''}${cssFontPx()}px AssPreview, sans-serif`;
      const advance = measureCanvas.measureText(text).width + state.spacingPx * text.length;
      return advance * state.scaleX / 100;
    }

    function breakCost(words, i) {
      const w = words[i];
      if (SENTENCE_END.test(w)) return -0.6;
      if (CLAUSE_END.test(w)) return -0.4;
      if (NO_LINE_END.has(bareWord(w))) return 0.8;
      const next = words[i + 1];
      if (next && ['же', 'ли', 'бы'].includes(bareWord(next))) return 0.6;
      return 0;
    }
    function partitionCost(widths, words, breaks, maxW, balance, space) {
      const bounds = [0, ...breaks, words.length];
      const lw = [];
      for (let k = 0; k + 1 < bounds.length; k++) {
        let sum = 0;
        for (let i = bounds[k]; i < bounds[k + 1]; i++) sum += widths[i];
        lw.push(sum + space * Math.max(0, bounds[k + 1] - bounds[k] - 1));
      }
      const mean = lw.reduce((a, b) => a + b, 0) / lw.length;
      let cost = lw.reduce((a, w) => a + Math.abs(w - mean), 0) / Math.max(1, maxW);
      lw.forEach(w => { if (w > maxW) cost += 10 + 10 * (w - maxW) / maxW; });
      breaks.forEach(b => { cost += breakCost(words, b - 1); });
      if (balance === 'bottom_heavy' || balance === 'top_heavy') {
        for (let k = 0; k + 1 < lw.length; k++) {
          const excess = balance === 'bottom_heavy' ? lw[k] - lw[k + 1] : lw[k + 1] - lw[k];
          if (excess > 0) cost += 1.5 * excess / maxW;
        }
      }
      if (bounds.length > 2 && bounds[bounds.length - 1] - bounds[bounds.length - 2] === 1 && words.length > 2) cost += 0.5;
      return cost;
    }
    function* combinations(n, k, start = 1, acc = []) {
      if (acc.length === k) { yield acc.slice(); return; }
      for (let i = start; i < n; i++) { acc.push(i); yield* combinations(n, k, i + 1, acc); acc.pop(); }
    }
    function wrapWords(words, maxW, maxLines, balance) {
      if (!words.length) return [];
      const space = textWidth(' ');
      const widths = words.map(textWidth);
      const total = widths.reduce((a, b) => a + b, 0) + space * (words.length - 1);
      if (total <= maxW || maxLines <= 1 || words.length === 1) return [words.slice()];
      const apply = breaks => { const b = [0, ...breaks, words.length]; const out = []; for (let k = 0; k + 1 < b.length; k++) out.push(words.slice(b[k], b[k + 1])); return out; };
      if (balance === 'greedy') {
        const breaks = []; let line = widths[0];
        for (let i = 1; i < words.length; i++) {
          if (line + space + widths[i] > maxW && breaks.length < maxLines - 1) { breaks.push(i); line = widths[i]; }
          else line += space + widths[i];
        }
        return apply(breaks);
      }
      let best = null;
      for (let lines = 2; lines <= Math.min(maxLines, words.length); lines++) {
        let fit = null;
        for (const combo of combinations(words.length, lines - 1)) {
          const c = partitionCost(widths, words, combo, maxW, balance, space);
          if (!fit || c < fit[0]) fit = [c, combo];
        }
        if (!fit) continue;
        if (!best || fit[0] < best[0]) best = fit;
        if (fit[0] < 10) break;
      }
      return apply(best[1]);
    }

    // Words of the sample as the burner would show them.
    function displayWords() {
      const caseMode = cfgValue('cfgSubsCase', 'none');
      const punct = cfgValue('cfgSubsPunct', 'keep');
      return state.sampleText.trim().split(/\s+/).filter(Boolean)
        .map(w => applyCase(stripPunct(w, punct), caseMode)).filter(Boolean);
    }
    // Where to cut a cue in two (port of _best_split_index): near the middle,
    // at a natural point.
    function bestSplitIndex(words) {
      const total = words.reduce((a, w) => a + w.length, 0) || 1;
      let best = Math.floor(words.length / 2), bestCost = Infinity, left = 0;
      for (let i = 1; i < words.length; i++) {
        left += words[i - 1].length;
        let cost = Math.abs(2 * left - total) / total;
        const prev = words[i - 1];
        if (SENTENCE_END.test(prev)) cost -= 0.35;
        else if (CLAUSE_END.test(prev)) cost -= 0.2;
        if (NO_LINE_END.has(bareWord(prev))) cost += 0.4;
        if (cost < bestCost) { best = i; bestCost = cost; }
      }
      return best;
    }

    // The sample split into cues the way the burner splits a transcript:
    // max_words_per_cue first, then halve any cue that does not fit its lines.
    function buildCues(maxW, maxLines, balance) {
      const words = displayWords();
      const limit = parseInt(cfgValue('cfgSubsMaxWords', 0), 10) || 0;
      let cues = [];
      if (limit > 0) {
        let cur = [];
        words.forEach(w => {
          if (cur.length >= limit) { cues.push(cur); cur = []; }
          cur.push(w);
          if (SENTENCE_END.test(w)) { cues.push(cur); cur = []; }
        });
        if (cur.length) cues.push(cur);
      } else {
        cues = [words];
      }
      const fits = cue => {
        const rows = wrapWords(cue, maxW, maxLines, balance);
        return rows.length <= maxLines && rows.every(r => textWidth(r.join(' ')) <= maxW + 0.5);
      };
      const split = (cue, depth) => {
        if (cue.length <= 1 || depth >= 8 || fits(cue)) return [cue];
        const cut = bestSplitIndex(cue);
        return [...split(cue.slice(0, cut), depth + 1), ...split(cue.slice(cut), depth + 1)];
      };
      return cues.flatMap(c => split(c, 0)).filter(c => c.length);
    }

    function highlightMode() {
      const mode = cfgValue('cfgSubsHighlight', state.autoAnimate ? 'karaoke' : 'none');
      return mode || 'none';
    }

    // Repaint just the per-word states — the 400ms tick must not rebuild the
    // DOM, refetch the font and force a full relayout.
    function paintHighlight() {
      const words = document.querySelectorAll('#subContainer .ass-word');
      const mode = highlightMode();
      // RU: Со стоп-кадром показываем середину реплики: начало уже «спето», хвост
      //     ещё нет — так видны обе заливки и обе настраиваются.
      // EN: The frozen preview shows mid-cue: the head is already spoken, the
      //     tail is not, so both fills stay visible and tunable.
      const active = state.autoAnimate ? activeIdx : Math.min(words.length - 1, Math.ceil(words.length * 0.6) - 1);
      words.forEach((span, i) => {
        span.classList.remove('spoken', 'current', 'hidden', 'pop', 'hl');
        if (mode === 'none') { span.classList.add('spoken'); return; }
        if (mode === 'karaoke') { if (i < active) span.classList.add('spoken'); return; }
        const isCurrent = i === active;
        const isActive = isCurrent || (mode === 'fill' && i < active);
        if (mode === 'reveal' && i > active) span.classList.add('hidden');
        if (isActive) span.classList.add(state.hlEnabled ? 'hl' : 'current');
        else if (!state.hlEnabled && mode !== 'reveal') span.classList.add('inactive-sec');
        else span.classList.add('spoken');
        if (mode === 'pop' && isCurrent && state.autoAnimate) {
          void span.offsetWidth;  // restart the CSS animation
          span.classList.add('pop');
        }
      });
    }

    function strokeAndShadow(prefix) {
      const s = prefix === 'hl'
        ? { bs: state.hlBorderStyle, oc: state.hlOutlineColor, oo: state.hlOutlineOp, o: state.hlOutline, bc: state.hlBackColor, bo: state.hlBackOp, sh: state.hlShadow }
        : { bs: state.borderStyle, oc: state.outlineColor, oo: state.outlineOp, o: state.outline, bc: state.backColor, bo: state.backOp, sh: state.shadow };
      const blur = parseFloat(cfgValue('cfgSubsBlur', 0)) || 0;
      const oC = toRGBA(s.oc, s.oo), bC = toRGBA(s.bc, s.bo);
      const out = { stroke: '0', shadow: 'none', bg: 'transparent', pad: '0' };
      if (s.bs == 1) {
        const shadows = [];
        if (s.o > 0) out.stroke = `${s.o * 2}px ${oC}`;
        // \blur softens the outline into a glow.
        if (blur > 0 && s.o > 0) shadows.push(`0 0 ${blur * 2}px ${oC}`, `0 0 ${blur * 4}px ${oC}`);
        if (s.sh > 0) shadows.push(`${s.sh}px ${s.sh}px ${blur}px ${bC}`);
        out.shadow = shadows.join(', ') || 'none';
      } else if (s.bs == 3) {
        out.bg = bC; out.pad = `${Math.max(2, s.o)}px`;
        // BorderStyle 3 paints the box with OutlineColour; BackColour is the shadow.
        out.bg = toRGBA(s.oc, s.oo);
      } else if (s.bs == 4 && s.o > 0 && s.oo > 0) {
        out.stroke = `${s.o * 2}px ${oC}`;
      }
      return out;
    }

    function updatePreview() {
      const c = document.getElementById('subContainer');

      applyFontFace();

      c.style.setProperty('--css-font', "'AssPreview'");
      c.style.setProperty('--css-font-size', `${cssFontPx()}px`);
      c.style.setProperty('--css-spacing', `${state.spacingPx}px`);
      c.style.setProperty('--css-weight', state.bold ? 'bold' : 'normal');
      c.style.setProperty('--css-style', state.italic ? 'italic' : 'normal');

      let decor = [];
      if (state.underline) decor.push('underline');
      if (state.strikeout) decor.push('line-through');
      c.style.setProperty('--css-decoration', decor.join(' ') || 'none');

      c.style.setProperty('--css-color-pri', toRGBA(state.primaryColor, state.primaryOp));
      c.style.setProperty('--css-color-sec', toRGBA(state.secondaryColor, state.secondaryOp));
      c.style.setProperty('--css-color-hl', toRGBA(state.hlColor, state.hlOp));

      // subtitles.vertical_align overrides the style's row, keeping its column.
      let align = parseInt(state.alignment);
      const vAlign = cfgValue('cfgSubsVAlign', 'style');
      const vOffset = parseFloat(cfgValue('cfgSubsVOffset', 0)) || 0;
      let marginV = state.marginV;
      const col = (align - 1) % 3;
      const styleRow = align <= 3 ? 'bottom' : align <= 6 ? 'center' : 'top';
      if (vAlign !== 'style') {
        align = { bottom: 1, center: 4, top: 7 }[vAlign] + col;
        if (vAlign !== styleRow && vAlign !== 'center') marginV = vAlign === 'bottom' ? 470 : 250;
      }
      marginV += Math.round(vOffset * 1920);

      // Usable width: frame minus margins, capped (or widened) by max_width_ratio.
      const ratio = parseFloat(cfgValue('cfgSubsMaxWidth', 0.74)) || 0.74;
      let mL = state.marginL, mR = state.marginR;
      const target = ratio * 1080;
      if (target > 1080 - mL - mR && col === 1) { mL = mR = Math.round((1080 - target) / 2); }
      const maxW = Math.min(1080 - mL - mR, target);

      c.style.setProperty('--css-top', 'auto'); c.style.setProperty('--css-bottom', 'auto');
      c.style.setProperty('--css-left', 'auto'); c.style.setProperty('--css-right', 'auto');
      c.style.setProperty('--css-transform-container', 'none');

      if ([7,8,9].includes(align)) { c.style.setProperty('--css-top', marginV + 'px'); }
      else if ([4,5,6].includes(align)) {
        c.style.setProperty('--css-top', `calc(50% - ${Math.round(vOffset * 1920)}px)`);
        c.style.setProperty('--css-transform-container', 'translateY(-50%)');
      }
      else { c.style.setProperty('--css-bottom', marginV + 'px'); }

      if ([1,4,7].includes(align)) {
        c.style.setProperty('--css-left', mL + 'px'); c.style.setProperty('--css-align-items', 'flex-start'); c.style.setProperty('--css-text-align', 'left');
      } else if ([3,6,9].includes(align)) {
        c.style.setProperty('--css-right', mR + 'px'); c.style.setProperty('--css-align-items', 'flex-end'); c.style.setProperty('--css-text-align', 'right');
      } else {
        c.style.setProperty('--css-left', mL + 'px'); c.style.setProperty('--css-right', mR + 'px'); c.style.setProperty('--css-align-items', 'center'); c.style.setProperty('--css-text-align', 'center');
      }

      c.style.setProperty('--css-transform-word', `scale(${state.scaleX/100}, ${state.scaleY/100}) rotate(${-state.angle}deg)`);
      c.style.setProperty('--css-transform-hl', `scale(${state.scaleX/100 * state.hlScale/100}, ${state.scaleY/100 * state.hlScale/100}) rotate(${-state.angle}deg)`);

      const base = strokeAndShadow('base');
      c.style.setProperty('--css-text-stroke', base.stroke);
      c.style.setProperty('--css-text-shadow', base.shadow);
      // BorderStyle 3: a box per line; 4: one box around the whole cue.
      const lineBox = state.borderStyle == 3 ? toRGBA(state.outlineColor, state.outlineOp) : 'transparent';
      c.style.setProperty('--css-box-bg', lineBox);
      c.style.setProperty('--css-box-pad', state.borderStyle == 3 ? `${Math.max(2, state.outline)}px` : '0');
      c.style.setProperty('--css-cue-bg', state.borderStyle == 4 ? toRGBA(state.backColor, state.backOp) : 'transparent');
      c.style.setProperty('--css-cue-pad', state.borderStyle == 4 ? `${state.outline}px` : '0');
      if (state.borderStyle == 4 && state.outlineOp <= 0) c.style.setProperty('--css-text-stroke', '0');

      const hl = strokeAndShadow('hl');
      c.style.setProperty('--css-hl-stroke', hl.stroke);
      c.style.setProperty('--css-hl-shadow', hl.shadow);
      c.style.setProperty('--css-hl-bg', hl.bg);
      c.style.setProperty('--css-hl-pad', hl.pad === '0' ? '0' : `0 ${hl.pad}`);

      // Lay out the current cue with the burner's cue-split and line-break rules.
      const wrap = document.getElementById('cfgSubsWrap');
      const maxLines = (wrap && !wrap.checked) ? 1 : (parseInt(cfgValue('cfgSubsMaxLines', 2), 10) || 2);
      const balance = cfgValue('cfgSubsBalance', 'bottom_heavy');
      const innerW = maxW - 2 * state.outline;
      const cues = buildCues(innerW, maxLines, balance);
      const words = cues.length ? cues[(state.autoAnimate ? cueIdx : 0) % cues.length] : [];
      currentCueLength = words.length;
      const rows = wrapWords(words, innerW, maxLines, balance);

      c.innerHTML = '';
      const cueDiv = document.createElement('div');
      cueDiv.className = 'ass-cue';
      rows.forEach(row => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'ass-line';
        row.forEach((w, i) => {
          // RU: Реальный пробел между словами — как в прожиге (" ".join).
          // EN: A real space between words, matching the burner's " ".join.
          if (i > 0) lineDiv.appendChild(document.createTextNode(' '));
          const span = document.createElement('span'); span.className = 'ass-word';
          span.textContent = w; lineDiv.appendChild(span);
        });
        cueDiv.appendChild(lineDiv);
      });
      c.appendChild(cueDiv);
      paintHighlight();

      document.querySelectorAll('.full-preset-btn').forEach(b => b.classList.toggle('active', b.dataset.preset === state.activePreset));

      const platData = PLATFORMS[state.platform] || PLATFORMS.ig;
      (document.getElementById('assEditorRoot') || document.documentElement).style.setProperty('--brand-color', platData.color);
      document.getElementById('ambientGlow').style.background = platData.color;
      // Safe-zone caption is localised through the shared i18n bridge (falls back to the RU default).
      const specsKey = 'ed_specs_' + state.platform;
      document.getElementById('specs-text').innerHTML =
        (window.t && window.t(specsKey) !== specsKey) ? window.t(specsKey) : platData.desc;

      const uiLayer = document.getElementById('uiLayer');
      uiLayer.innerHTML = state.showUI ? platData.ui : '';

      document.getElementById('mask-polygon').setAttribute('points', platData.points);
      document.getElementById('safe-outline').setAttribute('points', platData.points);

      document.getElementById('darkness-rect').setAttribute('opacity', state.showMask ? state.maskOpacity / 100 : '0');
      document.getElementById('safe-outline').style.opacity = state.showOutline ? '1' : '0';

      document.querySelectorAll('.plat-btn').forEach(b => b.classList.remove('active'));
      const activePlatBtn = document.querySelector(`.plat-btn[data-platform="${state.platform}"]`);
      if (activePlatBtn) activePlatBtn.classList.add('active');
    }

    function fontName() {
      // The burner maps this file-stem name to the font's real family.
      return state.fontPath.split('/').pop().replace(/\.[^/.]+$/, "");
    }

    function styleLine(name, s) {
      return `Style: ${name},${fontName()},${s.fontSizePx},` +
             `${toASSColor(s.primaryColor, s.primaryOp)},` +
             `${toASSColor(s.secondaryColor, s.secondaryOp)},` +
             `${toASSColor(s.outlineColor, s.outlineOp)},` +
             `${toASSColor(s.backColor, s.backOp)},` +
             `${s.bold?-1:0},${s.italic?-1:0},${s.underline?-1:0},${s.strikeout?-1:0},` +
             `${+(s.scaleX).toFixed(2)},${+(s.scaleY).toFixed(2)},${s.spacingPx},${s.angle},` +
             `${s.borderStyle},${s.outline},${s.shadow},${s.alignment},` +
             `${s.marginL},${s.marginR},${s.marginV},1`;
    }

    function getASS() {
      const lines = [
        '[Script Info]', 'ScriptType: v4.00+', 'PlayResX: 1080', 'PlayResY: 1920',
        'WrapStyle: 0', 'ScaledBorderAndShadow: yes', '',
        '[V4+ Styles]',
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
        styleLine('Default', state),
      ];
      if (state.hlEnabled) {
        const k = state.hlScale / 100;
        lines.push(styleLine('Highlight', {
          ...state,
          primaryColor: state.hlColor, primaryOp: state.hlOp,
          secondaryColor: state.hlColor, secondaryOp: state.hlOp,
          borderStyle: state.hlBorderStyle,
          outlineColor: state.hlOutlineColor, outlineOp: state.hlOutlineOp, outline: state.hlOutline,
          backColor: state.hlBackColor, backOp: state.hlBackOp, shadow: state.hlShadow,
          scaleX: state.scaleX * k, scaleY: state.scaleY * k,
        }));
      }
      return lines.join('\n');
    }

    function sync() {
      Object.keys(DEFAULTS).forEach(id => {
        let el = document.getElementById(id);
        if(!el) return;
        
        if(el.type === 'checkbox') el.checked = state[id];
        else el.value = state[id];
        
        let valEl = document.getElementById('v-'+id);
        if(valEl) valEl.innerText = state[id];
      });
      
      updatePreview();
      document.getElementById('assOutput').value = getASS();
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }

    Object.keys(DEFAULTS).forEach(id => {
      let el = document.getElementById(id);
      if(!el) return;
      el.addEventListener('input', e => {
        if(el.type === 'checkbox') state[id] = e.target.checked;
        else if(el.type === 'range' || el.type === 'number') state[id] = parseFloat(e.target.value);
        else state[id] = e.target.value;
        sync();
      });
    });

    document.querySelectorAll('.plat-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        // currentTarget, not target: a click can land on a ripple/child node.
        state.platform = e.currentTarget.getAttribute('data-platform');
        sync();
      });
    });

    const fileInput = document.getElementById('media-upload');
    const imgPreview = document.getElementById('preview-img');
    const vidPreview = document.getElementById('preview-vid');
    const placeholder = document.getElementById('placeholder');
    
    fileInput.addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (!file) return;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(file);
      placeholder.style.display = 'none';

      if (file.type.startsWith('video/')) {
          imgPreview.style.display = 'none';
          vidPreview.src = objectUrl; vidPreview.style.display = 'block';
      } else if (file.type.startsWith('image/')) {
          vidPreview.style.display = 'none'; vidPreview.pause();
          imgPreview.src = objectUrl; imgPreview.style.display = 'block';
      }
    });

    document.getElementById('clearMediaBtn').addEventListener('click', function() {
      fileInput.value = '';
      imgPreview.style.display = 'none'; vidPreview.style.display = 'none'; vidPreview.pause();
      placeholder.style.display = 'block';
      if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    });

    window.addEventListener('resize', resizePreview);
    // The phone is scaled to fit its pane, which also changes when the sidebar
    // reflows (sections expand, the window is split). updatePreview() no longer
    // rescales on every repaint, so watch the pane itself instead.
    if (window.ResizeObserver && workspace) {
      new ResizeObserver(() => resizePreview()).observe(workspace);
    }
    // Re-render the JS-built parts (safe-zone caption) when the page language changes.
    window.addEventListener('forge:langchange', () => updatePreview());

    setInterval(() => {
      if(state.autoAnimate) {
        activeIdx += 1;
        if (activeIdx > currentCueLength) {
          // Next cue of the sample (max_words_per_cue splits it into several).
          activeIdx = 0; cueIdx += 1;
          updatePreview();
        } else {
          paintHighlight();
        }
      }
    }, 400);

    // Render settings owned by app.js also shape the preview.
    ['cfgSubsHighlight', 'cfgSubsCase', 'cfgSubsPunct', 'cfgSubsBalance', 'cfgSubsMaxWords',
     'cfgSubsMaxLines', 'cfgSubsMaxWidth', 'cfgSubsVAlign', 'cfgSubsVOffset', 'cfgSubsBlur',
     'cfgSubsWrap'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener(el.type === 'checkbox' ? 'change' : 'input', () => { activeIdx = 0; updatePreview(); });
    });
    // Font metrics change the line breaks once the font has actually loaded.
    if (document.fonts) document.fonts.addEventListener('loadingdone', () => {
      assScaleCache.key = '';  // the span was measured on the fallback font
      updatePreview();
      document.getElementById('assOutput').value = getASS();
    });
    buildPresetGrid();
    window.addEventListener('forge:langchange', () => buildPresetGrid());

    document.getElementById('connectBtn').addEventListener('click', async () => {
      try {
        projHandle = await window.showDirectoryPicker({ mode: "readwrite" });
        document.getElementById('connectBtn').innerText = projHandle.name + " \u2714";
      } catch(e) { alert(e); }
    });

    // Small i18n helper: use the shared bridge if present, else fall back to the RU literal.
    const T = (key, fallback) => (window.t && window.t(key) !== key) ? window.t(key) : fallback;

    document.getElementById('applyBtn').addEventListener('click', async () => {
      if (!projHandle) return alert(T('ed_pick_first', "Сначала выберите папку проекта!"));
      try {
        let dir = projHandle;
        for (let p of ["assets", "subtitles"]) dir = await dir.getDirectoryHandle(p, { create: true });
        let file = await dir.getFileHandle("forge_subtitles.ass", { create: true });
        let writable = await file.createWritable();
        await writable.write(getASS());
        await writable.close();
        
        let btn = document.getElementById('applyBtn');
        btn.innerText = T('ed_saved', "\u0421\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u043e! \u2714");
        btn.style.background = "var(--md-sys-color-primary-container)"; 
        btn.style.color = "var(--md-sys-color-on-primary-container)";
        setTimeout(() => { 
            btn.innerText = T('ed_save_ass', "Сохранить файл ASS");
            btn.style.background = "var(--md-sys-color-primary)"; 
            btn.style.color = "var(--md-sys-color-on-primary)";
        }, 2000);
      } catch(e) { alert(T('ed_save_error', "Ошибка: ") + e); }
    });

    // Initial sync and calculation
    setTimeout(() => { sync(); resizePreview(); }, 100);
  
  })();
