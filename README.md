# Voice AI Chatbot

A voice-enabled chatbot that uses speech recognition, LLM for responses, and text-to-speech for natural conversation.

This project also leverages OpenAI and audio chunking for near real-time transcription, enabling faster and more responsive voice interactions. If your system can handle fast responses from the deepseek model, you can apply similar chunking logic in `main.py` for even smoother real-time experiences.

## Features

- Real-time voice input using sounddevice
- Speech-to-text using OpenAI's Whisper
- LLM-powered responses using Ollama
- Natural voice output using ElevenLabs
- Conversation memory with context
- Debug logging for monitoring interactions

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- ElevenLabs API key
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd voice_ai_bot
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix/MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here  # Optional, defaults to a specific voice
```

## Model Options

The chatbot supports different LLM models through Ollama:

1. **deepseek-r1:7b** (Recommended)
   - Better quality responses
   - Includes thinking process for more coherent answers
   - Requires more system resources
   - Minimum 16GB RAM recommended

2. **tinyllama** (Alternative)
   - Lighter weight model
   - Faster responses
   - Suitable for low-end systems
   - Minimum 8GB RAM recommended

To switch models, modify the `model` parameter in `main.py`:
```python
self.chat_model = ChatOllama(
    model="deepseek-r1:7b",  # or "tinyllama"
    temperature=0.5,
    num_predict=-1,  # or 128 for tinyllama
    base_url="http://localhost:11434"
)
```

## Usage

1. Start Ollama server:
```bash
ollama serve
```

2. Run the chatbot:
```bash
python main.py
```

3. Press Enter to start recording, and Enter again to stop.

## Features

- **Voice Input**: Records your voice and converts it to text
- **LLM Processing**: Generates contextual responses
- **Voice Output**: Converts responses to natural speech
- **Conversation Memory**: Maintains context of last 3 interactions
- **Debug Logging**: Shows detailed information about the conversation flow

## Notes

- The chatbot maintains a window of the last 3 messages for context
- Debug output shows the full conversation flow
- ElevenLabs voice can be customized through the API key
- Model responses are cleaned to remove thinking process from output

## Troubleshooting

- If you get memory errors, try using the tinyllama model
- Ensure Ollama server is running before starting the chatbot
- Check your ElevenLabs API key if voice output isn't working
- Verify your microphone is properly connected and configured

## License

[Your License Here]

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://github.com/ollama/ollama)
- [ElevenLabs](https://elevenlabs.io/)
- [OpenAI Whisper](https://github.com/openai/whisper) 