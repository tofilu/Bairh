    let mode = "positions";          // Aktuelle Phase des Knopfs
        let words = [];                  // Die zerlegten Wörter (Tokens)
        const active = new Set();        // Indizes der Wörter, bei denen ein Emoji folgen soll

        const form = document.getElementById("emoji-form");
        const actionBtn = document.getElementById("action-btn");
        const okBtn = document.getElementById("ok-btn");
        const backBtn = document.getElementById("back-btn");
        const input = document.getElementById("input");
        const strip = document.getElementById("token-strip");
        const result = document.getElementById("result");

        fetch("/llm/status")
            .then(r => r.json())
            .then(status => {
                if (!status.available) {
                    const cb = document.getElementById("use-llm");
                    cb.checked = false;
                    cb.disabled = true;
                }
            });

        // Enter wählt immer "OK" aus
        document.addEventListener("keydown", function (event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                if (mode === "positions") {
                    okBtn.click();
                } else {
                    actionBtn.click();
                }
            }
        });

        // "Postitionen wählen" wird genutzt
        actionBtn.addEventListener("click", async function (e) {
            e.preventDefault();

            if (mode === "positions") {
                tokenize();
                if (words.length === 0) return;        // kein Text -> nichts tun
                showStrip();
                mode = "ok";
                okBtn.style.display = "none";
                backBtn.style.display = "";
                actionBtn.value = "OK";
            } else {
                result.textContent = "Emojis werden geladen ...";
                await generate();
                showTextarea();
                words = [];
                mode = "positions";
                okBtn.style.display = "";
                backBtn.style.display = "none";
                actionBtn.value = "Positionen wählen";
            }
        });

        // "OK" wird genutzt
        okBtn.addEventListener("click", async function (e) {
            e.preventDefault();
            result.textContent = "Emojis werden geladen ...";
            active.clear();
            await generate();
        });

        // "Zurück" wird genutzt, ohne zu generieren zurück ins Eingabefeld
        backBtn.addEventListener("click", function (e) {
            e.preventDefault();
            showTextarea();
            mode = "positions";
            okBtn.style.display = "";
            backBtn.style.display = "none";
            actionBtn.value = "Positionen wählen";
        });

        // textarea ausblenden, Wort-Leiste an ihrer Stelle zeigen
        function showStrip() {
            input.style.display = "none";
            strip.style.display = "block";
        }

        // Wort-Leiste ausblenden, textarea zurückholen
        function showTextarea() {
            strip.style.display = "none";
            input.style.display = "";
        }

        // Text in klickbare Wörter, Zeilenumrbrüche ("\n" als eigenes Token) und Lücken zerlegen 
        function tokenize() {
            active.clear();
            words = input.value.trim()
                .split(/(\n)/)
                .flatMap(part => part === "\n" ? ["\n"] : part.split(/\s+/).filter(Boolean));

            strip.innerHTML = "";
            words.forEach((word, i) => {
                if (word === "\n") {
                    strip.append(document.createElement("br"));   // Umbruch wieder einbauen in die Darstellung
                    return;                                       // kein "+" nach einem Umbruch
                }
                const wordSpan = document.createElement("span");
                wordSpan.textContent = word;
                strip.append(wordSpan);

                const gap = document.createElement("span");
                gap.className = "gap";
                gap.textContent = "+";
                gap.addEventListener("click", () => {
                    if (active.has(i)) { active.delete(i); gap.classList.remove("active"); }
                    else { active.add(i); gap.classList.add("active"); }
                });
                strip.append(gap);
            });
        }

        // Ruft pro Abschnitt das Backend auf und gibt die top 4 Emojis zurück
        async function fetchCandidates(text) {
            const response = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: new URLSearchParams({ input: text, use_llm: document.getElementById("use-llm").checked })
            });
            return await response.json();
        }

        // Schickt den alternativ gewählten als Feedback ans Backend (Kontext = der Abschnitt vor dem Emoji)
        async function sendFeedback(emoji, segment) {
            await fetch("/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: new URLSearchParams({ emoji: emoji, emoji_feedback: segment })
            });
        }

        // Baut einen Emoji mit Alternativen-Menü beim Hover
        function buildEmoji(candidates, segment) {
            const wrapper = document.createElement("span");
            wrapper.className = "emoji-wrapper";

            const emojiSpan = document.createElement("span");
            emojiSpan.className = "generated-emoji";
            emojiSpan.textContent = candidates[0].emoji;

            const menu = document.createElement("div");
            menu.className = "emoji-menu";

            candidates.slice(1).forEach(candidate => {
                const alt = document.createElement("span");
                alt.className = "emoji-alt";
                alt.textContent = candidate.emoji;

                alt.addEventListener("click", () => {
                    const chosen = alt.textContent;
                    alt.textContent = emojiSpan.textContent;
                    emojiSpan.textContent = chosen;
                    sendFeedback(chosen, segment);
                });

                menu.append(alt);
            });

            wrapper.append(emojiSpan, menu);
            wrapper.addEventListener("mouseenter", () => positionMenu(wrapper, menu));
            return wrapper;
        }

        // Guckt wo der Emoji in der Output-Box ist und passt entsprechend die Darstellung vom Alternativ-Menü an beim hovern darüber
        function positionMenu(wrapper, menu) {
            const pad = 8;
            const bounds = wrapper.closest(".right-container").getBoundingClientRect();

            // Standard zurücksetzen: zentriert, nach unten
            menu.style.top = "100%";
            menu.style.bottom = "auto";
            menu.style.transform = "translateX(-50%)";

            const rect = menu.getBoundingClientRect();   // Wo ist das Emoji im Verhältnis zur Box

            // Menü in das Kasteninnere schieben, falls es überstehen würde
            let shift = 0;
            if (rect.left < bounds.left + pad) shift += bounds.left + pad - rect.left;
            if (rect.right > bounds.right - pad) shift -= rect.right - (bounds.right - pad);
            if (shift !== 0) menu.style.transform = `translateX(calc(-50% + ${shift}px))`;

            // Menü nach oben klappen, falls unten kein Platz ist
            if (rect.bottom > bounds.bottom - pad) {
                menu.style.top = "auto";
                menu.style.bottom = "100%";
            }
        }

        // Abschnitte bauen und Emojis and die gewählten Positionen setzen
        async function generate() {

            const points = [...active].sort((a, b) => a - b);

            // Keine Positionen gewählt oder OK gedrückt: ganzer Text = ein Abschnitt
            if (points.length === 0) {
                if (words.length > 0) {
                    text = words.filter(w => w !== "\n").join(" ");
                } else {
                    text = input.value;
                }
                const candidates = await fetchCandidates(text);
                result.textContent = "";
                result.append(text);
                result.append(" ");
                result.append(buildEmoji(candidates, text));
                return;
            }

            const segmentAt = new Map();
            const tasks = [];
            let start = 0;
            for (const k of points) {
                // Umbrüche aus dem Kontext entfernen, sonst stünde "\n" im Suchtext fürs Backend
                const segment = words.slice(start, k + 1).filter(w => w !== "\n").join(" ");
                segmentAt.set(k, segment);
                tasks.push(fetchCandidates(segment).then(candidates => [k, candidates]));
                start = k + 1;
            }
            const candidatesAt = new Map(await Promise.all(tasks));

            result.textContent = "";
            for (let i = 0; i < words.length; i++) {
                if (words[i] === "\n") {
                    result.append(document.createElement("br"));   // Umbruch in der Ausgabe wieder einsetzen
                    continue;
                }
                result.append(words[i]);
                if (candidatesAt.has(i)) {
                    result.append(" ");
                    result.append(buildEmoji(candidatesAt.get(i), segmentAt.get(i)));
                }
                // Leerzeichen nur zwischen zwei Wörtern, nicht vor einem Umbruch
                if (i < words.length - 1 && words[i + 1] !== "\n") result.append(" ");
            }
        }