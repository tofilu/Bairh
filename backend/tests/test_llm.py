from unittest.mock import patch
from backend import main
from backend.main import ListEntry

candidates = [
    ListEntry(emoji="😂", description="lacht", score=0.9),
    ListEntry(emoji="😭", description="weint", score=0.5),
]


def test_llm_returns_valid_emoji():
    with patch("backend.main.ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "😂"}}
        result = main.generate_with_llm("test", candidates)
        assert result == "😂"


def test_llm_fallback_on_invalid_emoji():
    with patch("backend.main.ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "🚀"}}
        result = main.generate_with_llm("test", candidates)
        assert result == "😂"


def test_llm_fallback_on_exception():
    with patch("backend.main.ollama.chat") as mock_chat:
        mock_chat.side_effect = Exception("ollama not running")
        result = main.generate_with_llm("test", candidates)
        assert result == "😂"


def test_llm_passes_correct_prompt():
    with patch("backend.main.ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "😂"}}
        main.generate_with_llm("hallo welt", candidates)
        call_kwargs = mock_chat.call_args.kwargs
        assert "model" in call_kwargs
        assert call_kwargs["model"] == main.LLM_Model
        assert "hallo welt" in call_kwargs["messages"][0]["content"]
