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
    messages: Annotated[Sequence[HumanMessage | AIMessage], "Messages in the conversation"]
    next: str

class VoiceChatbot:
    def __init__(self):
        # Ollama chat model
        self.chat_model = ChatOllama(
            model="deepseek-r1:7b",
            temperature=0.5,
            num_predict=-1,
            base_url="http://localhost:11434"
        )
        self.system_message = SystemMessage(content=
            """
You are a friendly and helpful AI assistant.
Speak in a natural, conversational tone. Be understanding of non-native English speakers
and maintain context from previous conversations. Keep responses concise and engaging.
"""
        )
        # Initialize conversation state
        self.conversation_state = ConversationState(messages=[], next="generate")
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        def generate_node(state: ConversationState) -> ConversationState:
            # Get last 3 messages for context
            recent_messages = state["messages"][-3:] if len(state["messages"]) > 3 else state["messages"]
            msgs = [self.system_message] + recent_messages
            
            # Debug logging for context
            print("\n=== Debug: Current Context ===")
            print(f"Total messages in state: {len(state['messages'])}")
            print("Recent messages being used for context:")
            for i, msg in enumerate(recent_messages):
                msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                print(f"{i+1}. [{msg_type}] {msg.content}")
            print("===========================\n")
            
            ai_msg: AIMessage = self.chat_model.invoke(msgs)
            print(f"Debug: Ollama reply content: {ai_msg.content}")
            
            # Clean the response before storing in state
            cleaned_content = self._clean_response(ai_msg.content)
            cleaned_ai_msg = AIMessage(content=cleaned_content)
            
            # Debug logging for state update
            print("\n=== Debug: State Update ===")
            print(f"Adding cleaned AI message to state. Total messages will be: {len(state['messages']) + 1}")
            print(f"Cleaned content: {cleaned_content}")
            print("===========================\n")
            
            state["messages"].append(cleaned_ai_msg)
            return state

        g = StateGraph(ConversationState)
        g.add_node("generate", generate_node)
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

    def _clean_response(self, text: str) -> str:
        """Remove the <think> section from the response."""
        if "<think>" in text and "</think>" in text:
            # Split on </think> and take everything after it
            parts = text.split("</think>", 1)
            if len(parts) > 1:
                return parts[1].strip()
        return text.strip()

    def generate_response(self, user_text: str) -> str:
        # Debug logging for new conversation
        print("\n=== Debug: New Conversation ===")
        print(f"Starting new conversation with user message: {user_text}")
        print("===========================\n")
        
        # Add user message to existing state
        self.conversation_state["messages"].append(HumanMessage(content=user_text))
        
        # Run through the workflow to get response
        out = self.workflow.invoke(self.conversation_state)
        
        # Update the conversation state with the new state
        self.conversation_state = out
        
        # Debug logging for final state
        print("\n=== Debug: Final State ===")
        print(f"Total messages in final state: {len(out['messages'])}")
        print("All messages in conversation:")
        for i, msg in enumerate(out["messages"]):
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            print(f"{i+1}. [{msg_type}] {msg.content}")
        print("===========================\n")
        
        # Get the most recent AI message (already cleaned)
        ai_messages = [msg for msg in out["messages"] if isinstance(msg, AIMessage) and msg.content.strip()]
        if ai_messages:
            reply = ai_messages[-1].content
            print(f"Debug: Returning AI reply: {reply}")
            return reply
        else:
            print("Debug: No valid AI response found")
            return "I apologize, but I couldn't generate a proper response. Could you please try again?"

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

