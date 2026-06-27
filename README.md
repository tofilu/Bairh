# Emoji-Generator

1. Ollama installieren: https://ollama.com/download
2. Llama 3 8B-Modell herunterladen
```ollama pull llama3```

3. Ollama Server starten (muss im Hintergrund laufen):
```ollama serve```

4. Backend-Ordner öffnen
```cd backend```

5. Virtuelle Umgebung einrichten/aktivieren
6. Requirements installieren
```pip install -r requirements.txt```

7. Anwendung starten mit uvicorn
```uvicorn main:app --reload```

8. Seite öffnen auf http://127.0.0.1:8000/
