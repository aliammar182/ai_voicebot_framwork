# Voice AI Bot Framework

A powerful voice-based AI chatbot framework that combines speech recognition, natural language processing, and text-to-speech capabilities. This project serves as a starting framework for building voice-enabled AI applications.

## Features

- 🎤 Real-time voice recording and transcription using OpenAI's Whisper
- 🤖 Natural language processing powered by LangChain and Ollama
- 🗣️ High-quality text-to-speech synthesis using ElevenLabs
- 🔄 Conversational memory and context management with LangGraph
- 🎯 Modular and extensible architecture

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- ElevenLabs API key
- PyAudio (for audio input/output)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voice-ai-bot.git
cd voice-ai-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_preferred_voice_id
```

## Usage

1. Ensure Ollama is running locally (default port: 11434)
2. Run the bot:
```bash
python main.py
```

3. Press Enter to start recording, and press Enter again to stop
4. The bot will transcribe your speech, process it, and respond with voice

## Model Options

### Ollama Models
The default configuration uses the `tinyllama` model, but you can use any model available in Ollama for better performance:
- `llama2` - Better performance but requires more resources
- `mistral` - Good balance of performance and resource usage
- `codellama` - Specialized for code-related conversations
- `neural-chat` - Optimized for chat interactions

To use a different model, modify the `model` parameter in the `ChatOllama` initialization in `main.py`.

### Speech Recognition Options
While the default implementation uses OpenAI's Whisper, you can modify the code to use Google's Live Transcription for faster response times:
1. Add `google-cloud-speech` to requirements.txt
2. Set up Google Cloud credentials
3. Modify the `transcribe_audio` method to use Google's Speech-to-Text API

## Architecture

- **Voice Input**: Uses `sounddevice` for audio recording
- **Speech Recognition**: OpenAI's Whisper model for accurate transcription
- **Language Processing**: 
  - LangChain for conversation management
  - Ollama (tinyllama model) for local LLM processing
  - LangGraph for workflow management and state handling
- **Voice Output**: ElevenLabs for natural-sounding speech synthesis

## Memory Usage

The bot maintains conversation context through LangGraph's state management system, which stores:
- Previous messages in the conversation
- System prompts and context
- Conversation flow state

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://github.com/ollama/ollama)
- [ElevenLabs](https://elevenlabs.io/)
- [OpenAI Whisper](https://github.com/openai/whisper) 