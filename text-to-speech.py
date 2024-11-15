import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

def transcribe_audio(file_path, model_path="vosk-model-small-en-us-0.15"):
    # Load the Vosk model
    model = Model(model_path)
    
    # Open the audio file
    wf = wave.open(file_path, "rb")
    
    # Ensure compatibility of audio file format
    if wf.getframerate() not in [8000, 16000, 32000, 44100, 48000]:
        raise ValueError("Audio file must have a sample rate of 8000, 16000, 32000, 44100, or 48000 Hz.")
    if wf.getnchannels() != 1:
        raise ValueError("Audio file must be mono channel.")
    
    # Initialize the recognizer with the sample rate
    recognizer = KaldiRecognizer(model, wf.getframerate())
    
    # Read the audio file and transcribe
    transcription = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            transcription += json.loads(result)["text"] + " "
    
    # Get final text from recognizer
    transcription += json.loads(recognizer.FinalResult())["text"]
    return transcription

def convert_to_mono(input_file, output_file="converted_mono_audio.wav", target_rate=16000):
    # Load the audio file with pydub (supports m4a)
    audio = AudioSegment.from_file(input_file)
    # Convert to mono and set the sample rate
    audio = audio.set_channels(1).set_frame_rate(target_rate)
    # Export as wav
    audio.export(output_file, format="wav")
    print(f"Converted audio saved as {output_file} with sample rate {target_rate} Hz and mono channel.")
    return output_file

# Usage
input_file_path = "Voice 001.m4a"  # Replace with your .m4a or .wav file
converted_audio_path = convert_to_mono(input_file_path)
print("Transcription:", transcribe_audio(converted_audio_path))
