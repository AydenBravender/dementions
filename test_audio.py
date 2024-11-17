import sounddevice as sd
import wave
import numpy as np
import os
import time
from SITA import Sita

# Folder to store recordings
response_dir_name = "ResponseAudios"

# Function to check and create the directory
def check_dir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# Function to record audio
def record_audio(duration, samplerate=44100):
    print(f"Recording for {duration} seconds...")
    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        audio_data.append(indata.copy())

    with sd.InputStream(callback=callback, channels=1, samplerate=samplerate, dtype='float32'):
        sd.sleep(duration * 1000)  # Duration in milliseconds

    print("Recording stopped.")
    return np.concatenate(audio_data, axis=0)

# Function to save audio
def save_audio(audio_data, filename, samplerate=44100):
    filepath = os.path.join(response_dir_name, filename)

    # Normalize audio data to prevent clipping
    audio_array = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_array.tobytes())
    print(f"Saved {filename}")

# Main function to record 5 times and process
def main():
    check_dir(response_dir_name)

    for i in range(1, 6):
        audio_data = record_audio(15)  # Record for 15 seconds
        save_audio(audio_data, f"output{i}.wav")

    # Merge and analyze after recording 5 files
    merge_and_analyze()

# Function to merge WAV files and run SITA
def merge_and_analyze():
    input_files = [
        os.path.join(response_dir_name, f"output{i}.wav") for i in range(1, 6)
    ]

    output_file = os.path.join(response_dir_name, "MergedAudio.wav")
    combined_audio = []
    sample_rate = None

    for file in input_files:
        with wave.open(file, "rb") as wf:
            if sample_rate is None:
                sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            combined_audio.append(audio_data)

    merged_audio = np.concatenate(combined_audio)

    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(merged_audio.tobytes())

    print(f"Merged audio saved as {output_file}")

    # Analyze merged audio using SITA
    sita = Sita(output_file)
    speech_score, emotion_score = sita.score_pair
    outlier_sentences = sita.outlier_sentences

    print(f"Speech Score: {speech_score}")
    print(f"Emotion Score: {emotion_score}")
    print(f"Outlier Sentences: {outlier_sentences}")

if __name__ == "__main__":
    main()
