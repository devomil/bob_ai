import sounddevice as sd
import numpy as np
import speech_recognition as sr
import pyttsx3  # For text-to-speech

class VoiceCommandProcessor:
    """
    Handles voice input and processes it into text commands.
    """
    def __init__(self, mic_index=None, sample_rate=16000, duration=5):
        """
        Initialize the VoiceCommandProcessor.

        Args:
            mic_index (int): Index of the microphone to use. Default is None (system default).
            sample_rate (int): Audio sample rate. Default is 16000 Hz.
            duration (int): Duration of audio recording in seconds. Default is 5 seconds.
        """
        self.recognizer = sr.Recognizer()
        self.mic_index = mic_index
        self.sample_rate = sample_rate
        self.duration = duration

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init(driverName='sapi5')
        self.configure_tts()

    def configure_tts(self):
        """
        Configure text-to-speech settings like voice, speed, and volume.
        """
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[0].id)  # Use voices[1].id for a female voice
        self.tts_engine.setProperty('rate', 200)  # Adjust speaking rate
        self.tts_engine.setProperty('volume', 0.9)  # Adjust volume (0.0 to 1.0)

    def speak(self, message):
        """
        Use TTS to audibly speak the given message.
        Args:
            message (str): The message to speak.
        """
        print(f"TTS: {message}")
        self.tts_engine.say(message)
        self.tts_engine.runAndWait()

    def listen_for_command(self):
        """
        Capture audio from the microphone and convert it to text.

        Returns:
            str: The recognized command as text, or None if the command is not recognized.
        """
        try:
            print("Listening for voice command...")
            # Record audio using sounddevice
            audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.mic_index,
            )
            sd.wait()  # Wait until recording is finished
            audio_data = np.int16(audio_data * 32767)  # Convert to 16-bit PCM
            
            # Convert audio to bytes
            audio_bytes = audio_data.tobytes()

            # Convert audio to a format usable by speech_recognition
            audio_source = sr.AudioData(audio_bytes, self.sample_rate, 2)

            # Recognize speech using Google Speech API
            command = self.recognizer.recognize_google(audio_source)
            print(f"Command recognized: {command}")
            return command
        except sr.UnknownValueError:
            self.speak("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError as e:
            self.speak(f"Error with the speech recognition service: {e}")
            return None
        except Exception as e:
            self.speak(f"Error during recording: {e}")
            return None


if __name__ == "__main__":
    """
    Standalone test mode for the VoiceCommandProcessor.
    """
    # Initialize VoiceCommandProcessor with default settings
    voice_processor = VoiceCommandProcessor()

    # Test loop
    while True:
        print("Say something or type 'exit' to quit.")
        command = voice_processor.listen_for_command()
        if command and command.lower() == 'exit':
            voice_processor.speak("Exiting the program. Goodbye!")
            break
        elif command:
            voice_processor.speak(f"You said: {command}")
