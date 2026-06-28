# Emoji-Generator

1. Ollama installieren: https://ollama.com/download
2. Llama 3 8B-Modell herunterladen
```ollama pull llama3```

alternativ mit `LLM_MODEL` ein anderes ollama-modell nehmen (z.B. `LLM_MODEL=llama3.1:8b`)

3. muss immer im Hintergrund laufen

4. Backend-Ordner öffnen
```cd backend```

5. *(Optional)* Virutelle Umgebuch einrichten/aktivieren
6. Requirements installieren
```pip install -r requirements.txt```

7. Anwendung starten mit uvicorn
```uvicorn main:app --reload```

wenn das sentence-transformer modell schon gecached ist, internetzugriff mit `HF_HUB_OFFLINE=1` vermeiden:
```HF_HUB_OFFLINE=1 uvicorn main:app --reload```

8. Seite öffnen auf http://127.0.0.1:8000/

9. *(Optional) Unter http://127.0.0.1:8000/debug/feedback kann man sich sein Feedback angucken (Also die Datenbank)*
