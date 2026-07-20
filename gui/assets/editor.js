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
      sampleText: "Разбираемся, почему этот выпуск вызывает споры.", autoAnimate: true,
      platform: "ig", showUI: true, showMask: true, showOutline: true, maskOpacity: 60
    };

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
        box: { borderStyle: 3, outline: 0, outlineColor: "#000000", outlineOp: 0.0, backColor: "#000000", backOp: 0.9, shadow: 0 },
        // Плашка + белый контур
        box_outline: { borderStyle: 3, outline: 2, outlineColor: "#FFFFFF", outlineOp: 1.0, backColor: "#000000", backOp: 0.85, shadow: 0 },
        // Неоновое свечение
        neon: { borderStyle: 1, outline: 4, outlineColor: "#000000", outlineOp: 1.0, shadow: 10, backColor: "#00F2EA", backOp: 0.9 }
      },
      geometry: {
        reels: { alignment: 2, marginV: 470 },
        shorts: { alignment: 2, marginV: 360 },
        center: { alignment: 5, marginV: 0 },
        top: { alignment: 8, marginV: 250 },
        tiktok: { alignment: 2, marginV: 560 }
      }
    };

    const state = { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") };
    let projHandle = null, activeIdx = 0, objectUrl = null;

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

    // Repaint just the karaoke highlight — the 400ms tick must not rebuild the
    // DOM, refetch the font and force a full relayout.
    function paintKaraoke() {
      const words = document.querySelectorAll('#subContainer .ass-word');
      // RU: Со стоп-кадром показываем середину \kf-свипа: начало строки уже
      //     «спето» (PrimaryColour), хвост ещё нет (SecondaryColour). Так обе
      //     заливки видны сразу и обе настраиваются — иначе одна из них молча
      //     ни на что не влияет.
      // EN: Frozen preview shows the middle of a \kf sweep: the head of the line
      //     is already "sung" (PrimaryColour), the tail is not (SecondaryColour).
      //     Both fills stay visible and tunable — otherwise one of them silently
      //     does nothing.
      const cut = state.autoAnimate ? activeIdx : Math.ceil(words.length * 0.6);
      words.forEach((span, i) => span.classList.toggle('active', i < cut));
    }

    function updatePreview() {
      const c = document.getElementById('subContainer');

      applyFontFace();

      c.style.setProperty('--css-font', "'AssPreview'");
      c.style.setProperty('--css-font-size', `${state.fontSizePx}px`);
      c.style.setProperty('--css-spacing', `${state.spacingPx}px`);
      c.style.setProperty('--css-weight', state.bold ? 'bold' : 'normal');
      c.style.setProperty('--css-style', state.italic ? 'italic' : 'normal');
      
      let decor = [];
      if (state.underline) decor.push('underline');
      if (state.strikeout) decor.push('line-through');
      c.style.setProperty('--css-decoration', decor.join(' ') || 'none');

      c.style.setProperty('--css-color-pri', toRGBA(state.primaryColor, state.primaryOp));
      c.style.setProperty('--css-color-sec', toRGBA(state.secondaryColor, state.secondaryOp));

      let align = parseInt(state.alignment);
      let mV = state.marginV + "px", mL = state.marginL + "px", mR = state.marginR + "px";
      
      c.style.setProperty('--css-top', 'auto'); c.style.setProperty('--css-bottom', 'auto');
      c.style.setProperty('--css-left', 'auto'); c.style.setProperty('--css-right', 'auto');
      c.style.setProperty('--css-transform-container', 'none');

      if ([7,8,9].includes(align)) { c.style.setProperty('--css-top', mV); }
      else if ([4,5,6].includes(align)) { c.style.setProperty('--css-top', '50%'); c.style.setProperty('--css-transform-container', 'translateY(-50%)'); }
      else { c.style.setProperty('--css-bottom', mV); }

      if ([1,4,7].includes(align)) { 
        c.style.setProperty('--css-left', mL); c.style.setProperty('--css-align-items', 'flex-start'); c.style.setProperty('--css-text-align', 'left'); c.style.setProperty('--css-justify', 'flex-start');
      } else if ([3,6,9].includes(align)) { 
        c.style.setProperty('--css-right', mR); c.style.setProperty('--css-align-items', 'flex-end'); c.style.setProperty('--css-text-align', 'right'); c.style.setProperty('--css-justify', 'flex-end');
      } else { 
        c.style.setProperty('--css-left', mL); c.style.setProperty('--css-right', mR); c.style.setProperty('--css-align-items', 'center'); c.style.setProperty('--css-text-align', 'center'); c.style.setProperty('--css-justify', 'center');
      }

      c.style.setProperty('--css-transform-word', `scale(${state.scaleX/100}, ${state.scaleY/100}) rotate(${-state.angle}deg)`);

      let oC = toRGBA(state.outlineColor, state.outlineOp);
      let bC = toRGBA(state.backColor, state.backOp);
      let oPx = state.outline;
      let sPx = state.shadow;

      c.style.setProperty('--css-box-bg', 'transparent'); c.style.setProperty('--css-box-pad', '0');
      c.style.setProperty('--css-text-stroke', '0'); c.style.setProperty('--css-text-shadow', 'none');

      if (state.borderStyle == 1) { 
        let shadows = [];
        if (oPx > 0) c.style.setProperty('--css-text-stroke', `${oPx*2}px ${oC}`);
        if (sPx > 0) shadows.push(`${sPx}px ${sPx}px 0 ${bC}`);
        c.style.setProperty('--css-text-shadow', shadows.join(', ') || 'none');
      } else { 
        c.style.setProperty('--css-box-bg', bC);
        c.style.setProperty('--css-box-pad', `25px 40px`);
        if (oPx > 0) c.style.setProperty('--css-text-stroke', `${oPx*2}px ${oC}`);
      }

      let words = state.sampleText.trim().split(/\s+/).filter(Boolean);
      c.innerHTML = '';
      let lineDiv = document.createElement('div');
      lineDiv.className = 'ass-line';
      words.forEach((w, i) => {
        // RU: Реальный пробел между словами — как в прожиге (" ".join). Раньше
        //     слова были соседними flex-элементами без разделителя, и зазор давал
        //     только column-gap = Spacing, равный 0 по умолчанию: в предпросмотре
        //     текст слипался в «почемуэтотвыпуск».
        // EN: A real space between words, matching the burner's " ".join. They
        //     used to be adjacent flex items whose only separation was
        //     column-gap = Spacing (0 by default), so the preview glued words
        //     together into "почемуэтотвыпуск".
        if (i > 0) lineDiv.appendChild(document.createTextNode(' '));
        let span = document.createElement('span'); span.className = 'ass-word';
        span.textContent = w; lineDiv.appendChild(span);
      });
      c.appendChild(lineDiv);
      paintKaraoke();

      const platData = PLATFORMS[state.platform] || PLATFORMS.ig;
      document.getElementById('assEditorRoot').style.setProperty('--brand-color', platData.color);
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

    function getASS() {
      let fname = state.fontPath.split('/').pop().replace(/\.[^/.]+$/, "");
      let style = `Style: Default,${fname},${state.fontSizePx},` +
                  `${toASSColor(state.primaryColor, state.primaryOp)},` +
                  `${toASSColor(state.secondaryColor, state.secondaryOp)},` +
                  `${toASSColor(state.outlineColor, state.outlineOp)},` +
                  `${toASSColor(state.backColor, state.backOp)},` +
                  `${state.bold?-1:0},${state.italic?-1:0},${state.underline?-1:0},${state.strikeout?-1:0},` +
                  `${state.scaleX},${state.scaleY},${state.spacingPx},${state.angle},` +
                  `${state.borderStyle},${state.outline},${state.shadow},${state.alignment},` +
                  `${state.marginL},${state.marginR},${state.marginV},1`;
      return `[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n${style}`;
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
        let max = state.sampleText.trim().split(/\s+/).filter(Boolean).length;
        activeIdx = (activeIdx + 1) % (max + 1);
        paintKaraoke();
      }
    }, 400);

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
