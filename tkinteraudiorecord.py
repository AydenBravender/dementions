import sounddevice as sd
import wave
import numpy as np
import customtkinter as ctk
import threading
import os
import shutil
from SITA import Sita

class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.recording = False  # To track recording state
        self.audio_data = []  # To store recorded audio chunks
        self.fs = 44100  # Sample rate (standard for most audio systems)
        self.filecount = 1
        self.sita = None

        #clear files
        self.clean_up(response_dir_name)

        # Configure GUI
        root.title("Audio Recorder")
        root.geometry("400x300")

        # Start Recording Button
        self.start_button = ctk.CTkButton(root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=20)

        # Stop Recording Button
        self.stop_button = ctk.CTkButton(root, text="Stop Recording", command=self.stop_recording)
        self.stop_button.pack(pady=20)

        # Status Label
        self.status_label = ctk.CTkLabel(root, text="Status: Idle")
        self.status_label.pack(pady=20)
    
    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.audio_data = []  # Clear previous audio data
            self.status_label.configure(text="Status: Recording...")
            threading.Thread(target=self.record_audio, daemon=True).start()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.status_label.configure(text="Status: Stopped")
            self.save_audio(f"output{self.filecount}.wav")
            self.status_label.configure(text=f"Audio saved as output{self.filecount}.wav")
            self.filecount += 1
            if self.filecount > 5:
                self.filecount = 1
                #trigger function to merge all audio files and clear file
                self.finalize_responses()

    def merge_wav_files(self, input_files, output_file):
        combined_audio = []
        sample_rate = None
        folder_name = response_dir_name
        filepath = os.path.join(folder_name, output_file)  # Combine folder and filename

        for file in input_files:
            with wave.open(file, "rb") as wf:
                # Ensure sample rate consistency
                if sample_rate is None:
                    sample_rate = wf.getframerate()
                elif wf.getframerate() != sample_rate:
                    raise ValueError(f"Sample rate of {file} does not match {sample_rate}")
                # Read audio frames and append to the list
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                combined_audio.append(audio_data)
        # Concatenate all audio data
        merged_audio = np.concatenate(combined_audio)

        # Save the combined audio
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(merged_audio.tobytes())
        print(f"Merged audio saved as {output_file}")

    def finalize_responses(self):
        global speech_score
        global emotion_score
        global outlier_sentences

        folder_path = response_dir_name

        input_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".wav")
        ]

        # Ensure there are files to merge
        if not input_files:
            raise FileNotFoundError("No .wav files found to merge.")

        output_file = "MergedAudio.wav"
        self.merge_wav_files(input_files, output_file)
        
        output_file = os.path.join(folder_path, output_file)

        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not found: {output_file}")

        self.sita = Sita(output_file)

        #score_pair in format cohesiveness(speech_score), tone(emotion_score)
        speech_score, emotion_score = self.sita.score_pair
        outlier_sentences = self.sita.outlier_sentences #is an array

        print(speech_score, emotion_score, outlier_sentences)
        self.clean_up(folder_path)

    def clean_up(self, folder_path):
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


    def record_audio(self):
        """Continuously records audio until stopped."""
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")  # Debug any issues
            if self.recording:
                self.audio_data.append(indata.copy())

        # Record in floating-point format for higher fidelity
        with sd.InputStream(callback=callback, channels=1, samplerate=self.fs, dtype='float32'):
            while self.recording:
                sd.sleep(100)  # Non-blocking sleep to allow GUI updates

    def save_audio(self, filename):
        folder_name = response_dir_name
        filepath = os.path.join(folder_name, filename)  # Combine folder and filename

        # Ensure the folder exists
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Convert and save the audio
        audio_array = np.concatenate(self.audio_data, axis=0)
        # Normalize the audio to prevent clipping
        audio_array = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.fs)
            wf.writeframes(audio_array.tobytes())

#file os functions
response_dir_name = "ResponseAudios"

#VARIABLES FOR YOU TO USE
speech_score = None
emotion_score = None
outlier_sentences = None #IS AN ARRAY OF SENTENCES; all out of context sentences in the merged audio file (not accurate lol just wanted to include my tokenization thing)

def check_dir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(f"Folder '{folder_name}' exists.")
    return folder_name

if __name__ == "__main__":
    check_dir(response_dir_name)
    root = ctk.CTk()
    app = AudioRecorderApp(root)
    root.mainloop()
