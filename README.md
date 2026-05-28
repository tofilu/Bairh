# Emoji-Generator / Emoji Recommender

> Finde das perfekte Emoji zu jedem Satz – unterstützt durch semantische Suche (Embeddings) und optional verfeinert durch ein lokales KI-Modell (LLM).

## ⚡ Quickstart

```bash
# 1. Abhängigkeiten installieren
cd Bairh
python3 -m venv .venv           # Linux/macOS
# python -m venv .venv          # Windows
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r backend/requirements.txt

# 2. (Optional) LLM-Modell für bessere Ergebnisse
ollama pull qwen2.5:0.5b

# 3. (Optional) Ollama-Server starten (falls nicht als Dienst aktiv)
ollama serve

# 4. Starten
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Danach **http://localhost:8000** im Browser öffnen und losschreiben.

> **Hinweis:** Die CSV-Datei `backend/emoji_dataset.csv` muss vor dem ersten Start vorhanden sein (Kaggle: „Complete Unicode Emoji Dataset").

---

## 📋 Inhaltsverzeichnis

- [Überblick](#-überblick)
- [Architektur](#-architektur)
- [Technologien](#-technologien)
- [Projektstruktur](#-projektstruktur)
- [Voraussetzungen](#-voraussetzungen)
- [Setup-Anleitung](#-setup-anleitung)
  - [1. Abhängigkeiten installieren](#1-abhängigkeiten-installieren)
  - [2. Datenbasis einfügen](#2-datenbasis-einfügen)
  - [3. (Optional) Ollama-Modell bereitstellen](#3-optional-ollama-modell-bereitstellen)
  - [4. Backend starten](#4-backend-starten)
  - [5. Frontend öffnen](#5-frontend-öffnen)
- [Funktionsweise (technisch)](#-funktionsweise-technisch)
  - [Phase 1: Datenbank-Initialisierung](#phase-1-datenbank-initialisierung)
  - [Phase 2: Embedding-basierte Suche](#phase-2-embedding-basierte-suche)
  - [Phase 3: (Optional) LLM-Verfeinerung](#phase-3-optional-llm-verfeinerung)
  - [Phase 4: Ergebnis-Rückgabe](#phase-4-ergebnis-rückgabe)
- [API-Referenz](#-api-referenz)
  - [`GET /health`](#get-health)
  - [`POST /recommend`](#post-recommend)
- [Frontend-Dokumentation](#-frontend-dokumentation)
- [LLM-Integration im Detail](#-llm-integration-im-detail)
- [Fehlerbehandlung & Fallbacks](#-fehlerbehandlung--fallbacks)
- [Ressourcenverbrauch](#-ressourcenverbrauch)
- [Anpassungsmöglichkeiten](#-anpassungsmöglichkeiten)
- [Fehlerbehebung (Troubleshooting)](#-fehlerbehebung-troubleshooting)

---

## 🎯 Überblick

Die **Emoji Recommender**-Anwendung ist eine vollständig lokal lauffähige Web-App, die zu einem eingegebenen Satz das semantisch passendste Emoji vorschlägt. 

**Hauptmerkmale:**

- 🔍 **Semantische Suche** – nicht einfach nur Stichwort-Matching, sondern echte Bedeutungsähnlichkeit durch neuronalen Embedding-Vergleich
- 🤖 **Optionaler LLM-Feinschliff** – ein lokales KI-Modell (qwen2.5:0.5b) entscheidet bei Unsicherheit mit, ohne dass Daten das Gerät verlassen
- 🏠 **100 % lokal** – keine Cloud, kein API-Key, keine Internetverbindung nach dem Setup
- 💻 **CPU-only** – läuft auf jedem Rechner, auch ohne Grafikkarte
- 🪶 **Leichtgewichtig** – ca. 2 GB RAM mit LLM, ~500 MB ohne LLM

---

## 🏗️ Architektur

```
┌──────────────┐     HTTP/JSON     ┌──────────────────────────────────┐
│   Browser    │ ◄──────────────► │          FastAPI-Server          │
│  (Frontend)  │   localhost:8000  │         (Python 3.10+)          │
│  index.html  │                   │                                  │
└──────────────┘                   │  ┌────────────────────────────┐  │
                                   │  │   /recommend (POST)       │  │
                                   │  │   /health    (GET)        │  │
                                   │  │   /          (GET - UI)   │  │
                                   │  └────────────────────────────┘  │
                                   │              │                   │
                                   │     ┌────────┴────────┐         │
                                   │     │                 │         │
                                   │     ▼                 ▼         │
                                   │  ChromaDB         Ollama       │
                                   │  (Vektor-DB)      (optional)   │
                                   │  ┌─────────┐     ┌──────────┐  │
                                   │  │Embeddings│     │qwen2.5   │  │
                                   │  │ + Daten  │     │:0.5b     │  │
                                   │  └─────────┘     └──────────┘  │
                                   └──────────────────────────────────┘
```

**Datenfluss einer Anfrage:**

```
1. Nutzer gibt Satz ein (z. B. "Ich liebe Pizza")
2. Frontend sendet POST /recommend mit {"text": "Ich liebe Pizza"}
3. FastAPI embeded den Text per Sentence-Transformer
4. ChromaDB sucht die 3 ähnlichsten Emoji-Beschreibungen
5. FastAPI berechnet Scores aus den Distanzen
6. Falls Top-Score < 0.84 → Ollama wird gefragt (LLM-Verfeinerung)
7. Antwort (empfohlenes Emoji + Kandidaten) geht zurück ans Frontend
8. Frontend zeigt Ergebnis an
```

---

## 🔧 Technologien

| Technologie | Version | Zweck |
|-------------|---------|-------|
| **Python** | 3.10+ | Programmiersprache |
| **FastAPI** | 0.110+ | Webframework (ASGI) |
| **ChromaDB** | 0.5+ | Lokale Vektordatenbank |
| **Sentence-Transformers** | 3.0+ | Embedding-Modell (`paraphrase-multilingual-MiniLM-L12-v2`) |
| **Ollama** | 0.3+ (optional) | Lokaler LLM-Server |
| **qwen2.5:0.5b** | – (optional) | Leichtes LLM (0,5 Mrd. Parameter) |
| **Pandas** | 2.0+ | CSV-Verarbeitung |
| **Uvicorn** | 0.29+ | ASGI-Server |
| **HTML/CSS/JS** | Vanilla | Frontend (kein Framework) |

**Warum diese Wahl?**

- **ChromaDB** statt Pinecone/Weaviate → kein externer Dienst, keine Kosten, alles lokal
- **paraphrase-multilingual-MiniLM-L12-v2** statt OpenAI-Embeddings → mehrsprachig (DE/EN/FR/…), 420 MB Modell, CPU-tauglich
- **qwen2.5:0.5b** statt GPT-4 → 0,5 Mrd. Parameter, ~500 MB RAM, läuft auf jedem Rechner
- **Vanilla JS** statt React/Vue → null Build-Step, eine einzige HTML-Datei

---

## 📁 Projektstruktur

```
Bairh/                            # ← Wurzelverzeichnis des Projekts
│
├── backend/                      # Backend (Python/FastAPI)
│   ├── main.py                   #   Hauptanwendung mit allen Endpunkten
│   ├── requirements.txt          #   Python-Abhängigkeiten (pip)
│   └── emoji_dataset.csv         #   (MUSS SELBST HINTERLEGT WERDEN)
│                                 #   Kaggle: "Complete Unicode Emoji Dataset"
│
├── frontend/                     # Frontend (HTML/CSS/JS)
│   └── index.html                #   Single-File-Weboberfläche
│
├── emoji_db/                     # Wird von ChromaDB automatisch erstellt
│   └── chroma.sqlite3            #   Persistente Vektordatenbank
│
└── README.md                     # Diese Dokumentation
```

### Frontend-Struktur (zweispaltig)

Das Frontend (`frontend/index.html`) kombiniert das klassische zweispaltige Layout des **Emoji-Generators** mit der modernen API-Anbindung des Emoji Recommenders:

```
.page-wrapper (max 900px, zentriert)
├── h1.page-title              → "Emoji-Generator"
├── .main-container (flex, gap 5%)
│   ├── .left-container (42%)  → Eingabe
│   │   ├── textarea           → Texteingabe (mehrzeilig)
│   │   ├── .toggle-row        → LLM ein/aus
│   │   └── button             → "OK" (indigo #4f46e5)
│   └── .right-container (42%) → Ergebnis
│       ├── #error             → Fehleranzeige
│       └── #result            → dynamisch: Emoji + Kandidaten
└── .bottom-container           → Hinweise
```

**Design-Prägung (`asai_projekt`-Stil):**
- Schriftart: **Fira Code** (monospace) – statt System-Schrift
- Hintergrund: `#f5f5f5` – heller, cleaner Look
- Container: weiße Boxen mit `border-radius: 12px` und `box-shadow`
- Farbschema: Indigo (`#4f46e5`) für Button und Fokus-Rahmen
- Eingabe: **Textarea** (mehrzeilig) statt einfachem Input-Feld

**Funktionalität:**
- API-Aufruf an `/recommend` per `fetch()`
- Anzeige des empfohlenen Emojis (groß, 5rem)
- Top-3 Kandidaten mit Prozent-Scores
- LLM-Badge bei KI-Verfeinerung
- LLM-Toggle (Checkbox) zum Ein-/Ausschalten
- Loading-Spinner während der Anfrage

---

## 📦 Voraussetzungen

Bevor du beginnst, stelle sicher, dass Folgendes installiert ist:

- **Python 3.10 oder neuer**
  ```bash
  python3 --version          # Linux/macOS
  # python --version         # Windows
  # Python 3.10.0 oder höher
  ```
- **pip** (Python-Paketmanager)
  ```bash
  python3 -m pip --version   # Linux/macOS
  # python -m pip --version  # Windows
  ```
- **Ein moderner Webbrowser** (Chrome, Firefox, Edge, Safari)
- **(Optional) Ollama** – nur wenn die LLM-Verfeinerung genutzt werden soll
  ```bash
  # Installationsanleitung: https://ollama.com
  ollama --version
  ```

---

## 🚀 Setup-Anleitung

### 1. Abhängigkeiten installieren

Lege ein virtuelles Environment an und installiere die benötigten Pakete:

```bash
# Ins Projektverzeichnis wechseln
cd Bairh

# Virtuelles Environment erstellen (isoliert die Abhängigkeiten)
python3 -m venv .venv           # Linux/macOS
# python -m venv .venv          # Windows

# Environment aktivieren
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Abhängigkeiten installieren
pip install -r backend/requirements.txt
```

**📄 `requirements.txt` – Enthaltene Pakete:**

| Paket | Zweck |
|-------|-------|
| `fastapi` | Webframework für die API |
| `uvicorn[standard]` | ASGI-Server (startet FastAPI) |
| `chromadb` | Vektordatenbank (lokal, persistent) |
| `sentence-transformers` | Embedding-Modell (`paraphrase-multilingual-MiniLM-L12-v2`) |
| `pandas` | CSV-Parsing |
| `ollama` | Python-Bibliothek für Ollama-API (optional) |

### 2. Datenbasis einfügen

Die Anwendung benötigt eine CSV-Datei mit Emojis und deren Beschreibungen.  
**Quelle:** [Kaggle – Complete Unicode Emoji Dataset with Meaning](https://www.kaggle.com/datasets/eliasdabbas/emoji-dataset)

**Erwartetes Format:**

```csv
emoji,description
😀,E1.0 grinning face
😃,E0.6 grinning face with big eyes
...
```

**Wichtige Spalten:**
- `emoji` – Das Unicode-Emoji-Zeichen (z. B. `🔥`)
- `description` – Die textuelle Beschreibung (z. B. `E0.6 fire`)

**Schritt:**

```bash
# CSV-Datei ins backend/-Verzeichnis kopieren
cp /pfad/zu/deiner/emoji_dataset.csv backend/emoji_dataset.csv   # Linux/macOS
# copy C:\pfad\zu\emoji_dataset.csv backend\emoji_dataset.csv    # Windows
```

Die Anwendung bereinigt die Beschreibungen automatisch (entfernt den Unicode-Version-Präfix wie `E0.6` → `fire`).

### 3. (Optional) Ollama-Modell bereitstellen

Wenn du die LLM-Verfeinerungsfunktion nutzen möchtest:

```bash
# Ollama installieren (falls noch nicht geschehen):
#   https://ollama.com – Download für Linux/macOS/Windows

# Das Modell pullen (einmalig, ca. 500 MB Download):
ollama pull qwen2.5:0.5b

# Prüfen, ob das Modell verfügbar ist:
ollama list
# → qwen2.5:0.5b sollte in der Liste erscheinen

# Ollama-Server starten (falls nicht bereits als Dienst aktiv):
ollama serve
```

**Wichtig:** Der `ollama serve`-Befehl muss in einem **separaten Terminal** laufen, während die App gestartet ist. Alternativ kann Ollama als Hintergrunddienst installiert werden.

**Ohne Ollama:** Die Anwendung funktioniert auch komplett ohne LLM – sie loggt dann eine Warnung und verwendet die reinen Embedding-Ergebnisse.

### 4. Backend starten

```bash
# Sicherstellen, dass das virtuelle Environment aktiv ist
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Backend starten (mit Hot-Reload für Entwicklung)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Beim ersten Start passiert Folgendes:**
1. Das mehrsprachige Modell `paraphrase-multilingual-MiniLM-L12-v2` wird heruntergeladen (ca. 420 MB, einmalig, unterstützt Deutsch)
2. Die CSV wird eingelesen (3944 Emojis)
3. ChromaDB erstellt Embeddings für alle Beschreibungen (dauert ca. 30–60 Sekunden)
4. Die Embeddings werden in `emoji_db/chroma.sqlite3` gespeichert
5. Der Server ist bereit: `http://localhost:8000`

**Erwartete Konsolenausgabe:**
```
INFO:     Will watch for changes in these directories: ['.../Bairh']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Lade CSV: .../backend/emoji_dataset.csv
INFO:     CSV geladen: 3944 Zeilen
INFO:     Batch 0–100 von 3944 hinzugefügt
INFO:     Batch 100–200 von 3944 hinzugefügt
...
INFO:     Datenbank initialisiert: 3944 Einträge
```

**Bei Folgestarts** erkennt ChromaDB, dass die Daten bereits vorhanden sind, und überspringt die Initialisierung:
```
INFO:     Collection enthält bereits 3944 Einträge
```

### 5. Frontend öffnen

Das Frontend wird **automatisch vom Backend ausgeliefert**.  
Einfach im Browser öffnen:

```
http://localhost:8000
```

**Bedienung:**
1. Einen Satz ins Textfeld eingeben (z. B. `Ich bin so glücklich heute`)
2. Auf **"Empfehlen"** klicken oder Enter drücken
3. Das Ergebnis erscheint sofort:
   - Empfohlenes Emoji (groß dargestellt)
   - Top-3 Kandidaten mit Prozent-Scores
   - Hinweis, ob das LLM zur Verfeinerung genutzt wurde

---

## ⚙️ Funktionsweise (technisch)

### Phase 1: Datenbank-Initialisierung

Beim Start der Anwendung wird `_init_database()` aufgerufen:

1. **Prüfung:** Ist die ChromaDB-Collection bereits gefüllt? (`collection.count() > 0`)
2. **CSV-Validation:** Existiert die CSV-Datei? → Sonst Abbruch mit Fehlermeldung
3. **CSV-Einlesen:** Pandas liest `emoji_dataset.csv` (Spalten: `emoji`, `description`)
4. **Beschreibungen bereinigen:** `_clean_description()` entfernt den Unicode-Version-Präfix:
   - `"E0.6 pizza"` → `"pizza"`
   - `"E1.0 grinning face"` → `"grinning face"`
5. **ChromaDB-Befüllung:** Alle 3944 Einträge werden in Batches von 100 hinzugefügt.  
   Jeder Eintrag besteht aus:
   - `id`: Zeilenindex als String (z. B. `"42"`)
   - `document`: Die bereinigte Beschreibung (**wird geembedded**)
   - `metadata`: `{"emoji": "🔥"}` (wird später als Ergebnis verwendet)

### Phase 2: Embedding-basierte Suche

Bei einer Anfrage an `POST /recommend`:

1. **Embedding:** Der eingegebene Text wird mit `paraphrase-multilingual-MiniLM-L12-v2` in einen 384-dimensionalen Vektor umgewandelt
2. **ChromaDB-Query:** Die Collection sucht die 3 ähnlichsten Dokumente mittels Cosine-Distance

   ```python
   results = collection.query(
       query_texts=["Ich liebe Pizza"],
       n_results=3
   )
   # → Gibt documents, metadatas, distances zurück
   ```
3. **Distance → Score:** Die ChromaDB-Distanz wird in einen verständlichen Score umgerechnet:

   ```
   score = 1 / (1 + distance)
   ```
   - Distance = 0 → Score = 1.0 (perfekte Übereinstimmung)
   - Distance = 1 → Score = 0.5 (mittlere Ähnlichkeit)
   - Distance = 3 → Score = 0.25 (geringe Ähnlichkeit)

### Phase 3: (Optional) LLM-Verfeinerung

Wenn der beste Score **kleiner als 0,84** ist, wird das LLM hinzugezogen:

> **Warum 0,84?**  
> Ein Score über 0,84 bedeutet, dass die Embedding-Suche zuverlässig ist.  
> Erst bei niedrigeren Scores lohnt sich der zusätzliche LLM-Aufruf.  
> Das LLM hat ein Token-Limit von 10 (`num_predict`), damit die Antwort blitzschnell kommt.

**Ablauf:**

1. **Prompt bauen:** Die drei Kandidaten werden in einen Prompt formatiert
2. **Ollama-Aufruf:** `qwen2.5:0.5b` erhält den Prompt und soll NUR das Emoji-Zeichen antworten
3. **Antwort parsen:** `_extract_emoji()` extrahiert das Emoji aus der Antwort (falls das Modell mehr Text liefert)
4. **Ergebniszuordnung:**
   - Wählte das LLM einen der 3 Kandidaten → dieser wird Top-Ergebnis
   - Wählte das LLM ein anderes Emoji → wird als "(LLM-Vorschlag)" markiert
   - Ollama nicht erreichbar → Embedding-Ergebnis bleibt bestehen

### Phase 4: Ergebnis-Rückgabe

Das Backend liefert JSON zurück:

```json
{
  "recommended": "🍕",
  "all_candidates": [
    { "emoji": "🍕", "description": "pizza", "score": 0.6672 },
    { "emoji": "🍣", "description": "sushi", "score": 0.6281 },
    { "emoji": "🥖", "description": "baguette bread", "score": 0.6223 }
  ],
  "llm_used": true
}
```

---

## 📡 API-Referenz

### `GET /health`

Prüft, ob der Server läuft, und gibt den Status der Datenbank zurück.

**Beispiel:**

```bash
curl http://localhost:8000/health
```

**Response (200 OK):**

```json
{
  "status": "ok",
  "collection_size": 3944
}
```

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `status` | string | `"ok"` wenn der Server betriebsbereit ist |
| `collection_size` | int | Anzahl der Emojis in der Datenbank (sollte 3944 sein) |

### `POST /recommend`

Der Haupt-Endpunkt. Gibt zu einem Text die passendsten Emojis zurück.

**Request:**

```json
{
  "text": "Ich liebe Pizza"
}
```

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `text` | string | **Erforderlich.** Der Satz, zu dem ein Emoji gefunden werden soll. |

**Response (200 OK):**

```json
{
  "recommended": "🍕",
  "all_candidates": [
    {
      "emoji": "🍕",
      "description": "pizza",
      "score": 0.6672
    },
    {
      "emoji": "🍣",
      "description": "sushi",
      "score": 0.6281
    },
    {
      "emoji": "🥖",
      "description": "baguette bread",
      "score": 0.6223
    }
  ],
  "llm_used": true
}
```

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `recommended` | string | Das finale, empfohlene Emoji (ggf. vom LLM optimiert) |
| `all_candidates` | array | Die Top-3 Kandidaten mit Scores (absteigend sortiert) |
| `all_candidates[].emoji` | string | Das Emoji-Zeichen |
| `all_candidates[].description` | string | Die bereinigte Beschreibung aus der CSV |
| `all_candidates[].score` | float | Ähnlichkeitsscore (0.0–1.0) |
| `llm_used` | boolean | `true`, wenn das LLM zur Verfeinerung eingesetzt wurde |

**Fehlerfälle:**

```json
// Leere Eingabe
{
  "recommended": "🤷",
  "all_candidates": [],
  "llm_used": false
}
```

---

## 🖥️ Frontend-Dokumentation

Das Frontend ist eine **einzige HTML-Datei** (`frontend/index.html`) – bewusst ohne Framework, Build-Tool oder Abhängigkeiten.

### Aufbau

**HTML-Struktur (zweispaltiges Layout):**
```
.page-wrapper (max 900px, zentriert)
├── h1.page-title              → "Emoji-Generator"
├── .main-container (flex, 5% gap)
│   ├── .left-container (42%)  → Eingabebereich
│   │   ├── textarea           → Texteingabe (mehrzeilig)
│   │   ├── .toggle-row        → LLM ein/aus
│   │   └── button             → "OK"
│   └── .right-container (42%) → Ergebnisbereich
│       ├── #error             → Fehleranzeige
│       └── #result            → dynamisch: Emoji + Kandidaten
└── .bottom-container           → Hinweise
```

Das Layout ist vom **asai_projekt** (Bairh) übernommen: zwei gleich große, weiße Container nebeneinander, darunter ein Hinweis-Bereich. Die rechte Seite zeigt standardmäßig den Platzhalter-Text "Hier ist dein Satz mit passendem Emoji ...".

**CSS:**
- Zweispaltiges Flexbox-Layout (`.main-container`), max. 900px breit
- Schriftart **Fira Code** (monospace) – von `asai_projekt` übernommen
- Indigo Akzentfarbe (`#4f46e5`) für Button + Fokus-Rahmen + Toggle
- Container: weiße Boxen mit `border-radius: 12px` und `box-shadow: 0 2px 8px`
- Textarea statt Input-Feld (mehrzeilig, ohne Resize-Griff)
- Loading-Spinner (animierter Kreis) während der API-Anfrage
- LLM-Toggle als iOS-artiger Schalter (via CSS `appearance: none`)

**JavaScript:**

Die drei Kernfunktionen:

| Funktion | Aufgabe |
|----------|---------|
| `recommend()` | Sendet den Text per `fetch()` an `POST /recommend`, ruft bei Erfolg `displayResult()` auf |
| `displayResult(data)` | Baut HTML aus den API-Daten (Emoji, Kandidaten, LLM-Badge) |
| `escapeHtml(str)` | Verhindert XSS, indem Sonderzeichen maskiert werden |

**Tastaturbedienung:**
- `Enter` im Textfeld löst die Suche aus (zusätzlich zum Button-Klick)

**Verhalten bei Fehlern:**
- Server nicht erreichbar → rote Fehlermeldung "Fehler: ..."
- HTTP-Fehler (z. B. 500) → Statuscode und Server-Antwort werden angezeigt
- Während der Anfrage → Button deaktiviert + Lade-Spinner

---

## 🤖 LLM-Integration im Detail

### Wann wird das LLM aktiviert?

Das LLM wird **nur unter bestimmten Bedingungen** aktiviert:

```
Wenn Top-Score < 0,84:
    └─ Wenn Ollama erreichbar:
        └─ LLM wählt aus den 3 Kandidaten
    └─ Sonst:
        └─ Embedding-Ergebnis bleibt (mit Log-Warnung)
Sonst (Score ≥ 0,84):
    └─ LLM wird nicht angefragt (Embedding reicht aus)
```

### Der Prompt

Der Prompt an das LLM ist bewusst einfach gehalten:

```
Wähle das passendste Emoji für: "Ich liebe Pizza".
Kandidaten: 🍕 (pizza); 🍣 (sushi); 🥖 (baguette bread)
Antworte NUR mit dem Emoji-Zeichen.
```

**Prompt-Design-Prinzipien:**
1. **Klare Aufgabenstellung:** "Wähle das passendste Emoji"
2. **Kontext:** Der ursprüngliche Nutzer-Text
3. **Eingeschränkte Auswahl:** Nur die 3 Kandidaten werden genannt
4. **Strikte Formatvorgabe:** "Antworte NUR mit dem Emoji-Zeichen"

### Antwortparsing

Da LLMs nicht immer strikten Anweisungen folgen, wird die Antwort mit `_extract_emoji()` geparst. Die Funktion sucht mit einem Regex nach dem ersten Emoji-Zeichen im Antwort-String – selbst wenn das Modell zusätzlichen Text liefert.

**Beispiele für LLM-Antworten und Parsing:**

| LLM-Antwort | Extrahiertes Emoji |
|-------------|-------------------|
| `🍕` | `🍕` |
| `🍕 ist am besten` | `🍕` |
| `Ich denke 🍕 passt am besten.` | `🍕` |
| `Keines der drei` | `None` (Fallback auf Embedding) |

---

## ⚠️ Fehlerbehandlung & Fallbacks

Die Anwendung ist darauf ausgelegt, auch bei Fehlern robust zu laufen.

| Fehlerszenario | Verhalten |
|----------------|-----------|
| **CSV fehlt** | Backend startet nicht; Fehlermeldung: "CSV-Datei nicht gefunden: ..." |
| **CSV falsches Format** | Pandas-Error (Spalten fehlen) → Startup schlägt fehl |
| **Ollama nicht installiert** | Log-Warnung; Embedding-Ergebnis wird verwendet |
| **Ollama installiert, Modell fehlt** | Log-Warnung; Embedding-Ergebnis wird verwendet |
| **Ollama läuft nicht** | Log-Warnung; Embedding-Ergebnis wird verwendet |
| **ChromaDB-Korruption** | `emoji_db/` löschen → Neubau beim nächsten Start |
| **Leere Eingabe** | Antwortet mit `🤷` und leerer Kandidatenliste |
| **Server überlastet** | Uvicorn verarbeitet Anfragen asynchron; ChromaDB ist Single-Threaded |

---

## 💾 Ressourcenverbrauch

| Komponente | RAM | Festplatte | CPU |
|------------|-----|------------|-----|
| FastAPI + Uvicorn | ~50 MB | – | < 1 % |
| ChromaDB | ~200 MB | ~5 MB (Datenbank) | Beim Query: < 5 % |
| Sentence-Transformer (geladen) | ~250 MB | ~420 MB (Modell, einmalig) | Beim Embedding: hoch, aber nur Startup |
| **Summe (ohne LLM)** | **~500 MB** | **~430 MB** | – |
| Ollama (qwen2.5:0.5b, geladen) | ~500 MB | ~500 MB (Modell) | Beim Prompt: ~20–50 % (CPU) |
| **Summe (mit LLM)** | **~850 MB** | **~585 MB** | – |

**Empfohlene Mindestanforderungen:**
- **Ohne LLM:** 2 GB RAM (läuft auf jedem Rechner)
- **Mit LLM:** 4 GB RAM (läuft auch auf älteren Laptops)
- **GPU:** Nicht erforderlich

---

## 🔧 Anpassungsmöglichkeiten

### Anderes Embedding-Modell

In `backend/main.py`:

```python
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # ← Hier tauschen
)
```

Alternativen (alle CPU-tauglich):
- `"all-MiniLM-L6-v2"` – Englisch only, 80 MB, schneller aber kein Deutsch
- `"all-MiniLM-L12-v2"` – Englisch only, 120 MB, genauer
- `"distiluse-base-multilingual-cased-v2"` – Mehrsprachig, 500 MB, oft genauer

### Anderes LLM

```python
response = ollama.chat(
    model="qwen2.5:0.5b",  # ← Hier tauschen
    messages=[...],
)
```

Alternativen (lokal, CPU-tauglich):
- `"llama3.2:1b"` – 1 Mrd. Parameter, besser aber ~1 GB RAM
- `"smollm2:360m"` – Noch kleiner (360 Mio.), schwächer
- `"gemma2:2b"` – 2 Mrd. Parameter, braucht ~2 GB RAM

### Score-Schwellwert anpassen

```python
if top.score < 0.84:  # ← Hier den Wert ändern (z. B. 0.5, 0.9)
```

- **Niedriger** (z. B. 0.5): LLM wird seltener aktiviert (mehr Vertrauen in Embeddings)
- **Höher** (z. B. 0.95): LLM wird öfter aktiviert (strengere Anforderungen)

### Anzahl der Kandidaten

```python
results = collection.query(
    query_texts=[req.text],
    n_results=5,  # ← Hier erhöhen/verringern
)
```

---

## 🔍 Fehlerbehebung (Troubleshooting)

### "Address already in use"

```bash
# Prüfen, ob der Port belegt ist
lsof -i :8000                         # Linux/macOS
# netstat -ano | findstr :8000       # Windows

# Prozess beenden
kill -9 <PID>                         # Linux/macOS
# taskkill /F /PID <PID>             # Windows

# oder alternativ anderen Port verwenden:
uvicorn backend.main:app --reload --port 8001
```

### "ModuleNotFoundError"

```bash
# Sicherstellen, dass das venv aktiv ist
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Abhängigkeiten (neu) installieren
pip install -r backend/requirements.txt
```

### ChromaDB-Fehler oder inkonsistente Daten

```bash
# Einfach den Datenbank-Ordner löschen → wird beim nächsten Start neu aufgebaut
rm -rf emoji_db/                    # Linux/macOS
# rmdir /s emoji_db                 # Windows
```

### Ollama-Fehler

```bash
# Prüfen, ob Ollama läuft
ollama list

# Ollama starten (falls nicht als Dienst läuft)
ollama serve

# Modell prüfen
ollama pull qwen2.5:0.5b
```

### "No module named 'torch'" (Sentence-Transformer)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
# CPU-only-Version von PyTorch (deutlich kleiner als die CUDA-Version)
```

---

## 📄 Lizenz

Dieses Projekt ist als Vorlage/Rahmencode bereitgestellt.  
Das verwendete Dataset unterliegt den Lizenzbedingungen von Kaggle und dem jeweiligen Dataset-Autor.

---

*Erstellt mit Python, FastAPI, ChromaDB und Sentence-Transformers.*
