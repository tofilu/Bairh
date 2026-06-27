# Emoji-Generator

1. **llama.cpp** herunterladen: https://github.com/ggerganov/llama.cpp/releases
   - **Windows**: `llama-bins-win-*.zip` (darin `llama-server.exe`)
   - **macOS**: `llama-bins-macos-*.zip` (darin `llama-server`)
   - **Linux**: `llama-bins-ubuntu-*.zip` (darin `llama-server`)

2. Ein GGUF-Modell herunterladen (z. B. Llama 3.2 3B):
   ```
   https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
   ```

3. llama.cpp Server starten (Port 8080, muss im Hintergrund laufen):
   ```
   ./llama-server -m Llama-3.2-3B-Instruct-Q4_K_M.gguf
   ```

4. Backend-Ordner öffnen
   ```bash
   cd backend
   ```

5. Virtuelle Umgebung einrichten/aktivieren
6. Requirements installieren
   ```bash
   pip install -r requirements.txt
   ```

7. Anwendung starten mit uvicorn
   ```bash
   uvicorn main:app --reload
   ```

8. Seite öffnen auf http://127.0.0.1:8000/
