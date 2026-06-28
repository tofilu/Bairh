# Emoji-Generator

1. Ollama installieren: https://ollama.com/download
2. Llama 3 8B-Modell herunterladen:
```ollama pull llama3```

>Alternativ mit `LLM_MODEL` ein anderes ollama-modell nehmen (z.B. `LLM_MODEL=llama3.1:8b`)

3. Ollama muss im Hintergund immer aktiv sein - standardmäßig ist ein Autostart hinterlegt. Ggf. manuell starten mit: 
```ollama serve```

4. Heruntergeladenes Projekt starten und Backend-Ordner öffnen:
```cd \dein PATH zum Projekt\backend```, bei UNIX: ```cd /dein PATH zum Projekt/backend```

>5. *(Optional)* Virutelle Umgebuch einrichten/aktivieren
6. Requirements installieren: 
```pip install -r requirements.txt```

7. Anwendung starten mit uvicorn: 
```uvicorn main:app```

>Wenn das sentence-transformer modell schon gecached ist, internetzugriff mit `HF_HUB_OFFLINE=1` vermeiden:
```$env:HF_HUB_OFFLINE=1; uvicorn main:app```, bei UNIX mit: ```HF_HUB_OFFLINE=1 uvicorn main:app```

8. Seite öffnen auf http://127.0.0.1:8000/

>*(Optional)* Unter http://127.0.0.1:8000/debug/feedback kann man sich das Feedback zur Datenbank angucken, wenn man andere Emojis ausgewählt hat.