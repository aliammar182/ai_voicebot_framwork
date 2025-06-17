import os
import types
import io
import sounddevice as sd
import numpy as np
import whisper
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, Sequence
from pydub import AudioSegment
from pydub.playback import play as play_audio
from elevenlabs.client import ElevenLabs
from scipy.io.wavfile import write as wav_write
import tempfile

# Load environment variables
load_dotenv()

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# ElevenLabs client setup
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv(
    "ELEVENLABS_VOICE_ID",
    "ErXwobaYiN019PkySvjV"  # default voice ID
)
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Define LangGraph state type
class ConversationState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], "Messages in the conversation"]
    next: str

class VoiceChatbot:
    def __init__(self):
        # Ollama chat model
        self.chat_model = ChatOllama(
            model="tinyllama",
            temperature=0.5,
            num_predict=128,
            base_url="http://localhost:11434"
        )
        self.system_message = SystemMessage(content=
            """
You are a friendly and helpful AI assistant.
Speak in a natural, conversational tone. Be understanding of non-native English speakers
and maintain context from previous conversations. Keep responses concise and engaging.
"""
        )
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        def generate_node(state: ConversationState) -> ConversationState:
            msgs = [self.system_message] + state["messages"]
            ai_msg: AIMessage = self.chat_model.invoke(msgs)
            print(f"Debug: Ollama reply content: {ai_msg.content}")
            state["messages"].append(ai_msg)
            state["next"] = "should_continue"
            return state

        def should_continue(state: ConversationState) -> ConversationState:
            state["next"] = "generate"
            return state

        g = StateGraph(ConversationState)
        g.add_node("generate", generate_node)
        g.add_node("should_continue", should_continue)
        g.add_edge("generate", "should_continue")
        g.add_conditional_edges("should_continue", {"generate": generate_node})
        g.set_entry_point("generate")
        return g.compile()

    def record_audio(self) -> np.ndarray:
        print("Press Enter to start recording, and press Enter again to stop.")
        input("Press Enter to start...")
        print("Recording... Press Enter to stop.")
        frames = []
        def cb(indata, fcount, time, status):
            if status:
                print(status)
            frames.append(indata.copy())
        with sd.InputStream(callback=cb, channels=1, samplerate=16000):
            input()
        print("Recording stopped.")
        return np.concatenate(frames, axis=0)

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            fname = f.name
            wav_write(fname, 16000, audio_data)
        res = whisper_model.transcribe(fname)
        os.remove(fname)
        return res.get("text", "")

    def generate_response(self, user_text: str) -> str:
        # Run through the workflow to get multiple chunks
        state = ConversationState(messages=[HumanMessage(content=user_text)], next="generate")
        out = self.workflow.invoke(state, config={"recursion_limit": 3})
        # Combine all AIMessage contents into one string
        ai_parts = [msg.content for msg in out["messages"] if isinstance(msg, AIMessage)]
        reply = "\n".join(ai_parts)
        print(f"Debug: Combined reply returned: {reply}")
        return reply

    def text_to_speech(self, text: str):
        # Request PCM audio
        raw = client.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id="eleven_monolingual_v1",
            output_format="pcm_16000"
        )
        audio_bytes = b"".join(raw) if isinstance(raw, types.GeneratorType) else raw
        segment = AudioSegment(
            data=audio_bytes,
            sample_width=2,
            frame_rate=16000,
            channels=1
        )
        play_audio(segment)

    def run(self):
        print("Voice Chatbot initialized. Press Ctrl+C to exit.")
        try:
            while True:
                audio = self.record_audio()
                text = self.transcribe_audio(audio)
                print(f"You said: {text}")
                reply = self.generate_response(text)
                print(f"Bot: {reply}")
                self.text_to_speech(reply)
        except KeyboardInterrupt:
            print("\nGoodbye!")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    bot = VoiceChatbot()
    bot.run()

