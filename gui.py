import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
from tkinter import Tk, Canvas
from tkinter import filedialog
import os
from model_clock import ImageClassifier
import cv2
import math
import mediapipe as mp
import time
from PIL import Image, ImageTk
import pygame
import customtkinter as ctk
from pygame.locals import *
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import time
import os
import sounddevice as sd
import wave
import numpy as np
import os
import time
from SITA import Sita
import threading
import random

predicted_class = 0
stage = 0
results_fit = 0
speech_score = 0
emotion_score = 0
elapsed_time = 0
# global predicted_class
# global stage
# global results_fit
# global speech_score
# global emotion_score
# global elapsed_time


time_start_end = []
# Initialize the main app window
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("CustomTkinter GUI")
app.geometry("800x800")

# Configure app layout to support dynamic resizing
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)

# Function to switch frames
def switch_frame(frame):
    frame.tkraise()

# Clear window function
def clear_window():
    for widget in app.winfo_children():
        widget.destroy()

    # Add two new buttons after clearing
    new_frame = ctk.CTkFrame(app)
    new_frame.grid(row=0, column=0, sticky="nsew")

    for widget in app.winfo_children():
        widget.destroy()

    try:
        questions()  # Ensure this is called
    except Exception as e:
        print(f"Error in questions(): {e}")

    for widget in app.winfo_children():
        widget.destroy()
    # Create new widgets for drawing interface
    drawing_app = DrawingApp(app)
    drawing_app.pack(fill="both", expand=True)

def questions():
    from pydub import AudioSegment
    from pydub.utils import which
    import customtkinter as ctk

    AudioSegment.converter = which("ffmpeg")  # Explicitly set ffmpeg path
    print("Starting questions function...")  # Trace start of function

    # Folder to store recordings
    response_dir_name = "ResponseAudios"

    # Function to check and create the directory
    def check_dir(folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        print(f"Directory '{folder_name}' checked/created successfully.")
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
        print(f"Saved {filename} at {filepath}")

    # Function to play a question and record the response
    def play_and_record(question_file, output_file, duration=1, question_label=None):
        try:
            print(f"Playing {question_file}...")
            audio = AudioSegment.from_file(question_file)
            data = np.array(audio.get_array_of_samples())

            if audio.channels == 2:
                data = data.reshape((-1, 2))

            sd.play(data, samplerate=audio.frame_rate)
            sd.wait()  # Wait for the audio to finish
            print(f"Finished playing {question_file}")

            # Update status
            if question_label:
                question_label.configure(text="Recording your response...")

            audio_data = record_audio(duration)
            save_audio(audio_data, output_file)
            print(f"Recorded response saved as {output_file}")
        except FileNotFoundError as fnf_error:
            print(f"File not found: {fnf_error}")
        except Exception as e:
            print(f"Error in play_and_record(): {e}")

    # Function to merge WAV files and run SITA analysis
    def merge_and_analyze():
        global speech_score, emotion_score  # Declare global for updating
        try:
            print("Starting audio merge...")
            input_files = [
                os.path.join(response_dir_name, f"output{i}.wav") for i in range(1, 6)
            ]

            for i, file in enumerate(input_files, start=1):
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Expected audio file not found: {file}")
                print(f"Found input file {i}: {file}")

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

            print("Running SITA analysis...")
            sita = Sita(output_file)
            speech_score, emotion_score = sita.score_pair
            outlier_sentences = sita.outlier_sentences

            print(f"Speech Score: {speech_score}")
            print(f"Emotion Score: {emotion_score}")
            print(f"Outlier Sentences: {outlier_sentences}")
        except FileNotFoundError as fnf_error:
            print(f"Merge file error: {fnf_error}")
        except Exception as e:
            print(f"Error in merge_and_analyze(): {e}")

    # Messages to display for each question
    question_messages = [
        "What was your favourite childhood memory?",
        "Do you have any children?",
        "Where were you born?",
        "What is your favorite food?",
        "What is your favorite hobby?"
    ]

    try:
        check_dir(response_dir_name)

        # Create a frame for the questions and display area
        question_frame = ctk.CTkFrame(app)
        question_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Add a label to display the questions
        question_label = ctk.CTkLabel(question_frame, text="", font=("Arial", 18))
        question_label.pack(pady=20)

        question_files = [
            '/home/ayden/Desktop/aydenprj/nathacks/dementions/Q1.m4a',
            '/home/ayden/Desktop/aydenprj/nathacks/dementions/Q2.m4a',
            '/home/ayden/Desktop/aydenprj/nathacks/dementions/Q3.m4a',
            '/home/ayden/Desktop/aydenprj/nathacks/dementions/Q4.m4a',
            '/home/ayden/Desktop/aydenprj/nathacks/dementions/Q5.m4a'
        ]

        for i, (question_file, message) in enumerate(zip(question_files, question_messages), start=1):
            print(f"Displaying message for question {i}: {message}")
            question_label.configure(text=message)
            app.update_idletasks()  # Ensure the UI updates before recording
            play_and_record(question_file, f"output{i}.wav", question_label=question_label)

        print("All questions completed. Now merging audio files...")
        merge_and_analyze()
        print("Merge and analysis complete.")
    except Exception as e:
        print(f"An error occurred in questions(): {e}")
    finally:
        print("Questions function execution finished.")


class DrawingApp(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = ctk.CTkCanvas(self, bg="white")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            200, 20,  # x, y coordinates for the text position
            text="Please Draw a Clock",
            font=("Arial", 16, "bold"),
            fill="black"
        )

        # Variables for drawing state
        self.last_x, self.last_y = None, None

        # Binding events to handle drawing
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)

        # Submit button for saving or submitting the drawing
        self.submit_button = ctk.CTkButton(self, text="Submit", command=self.submit_drawing, width=200)
        self.submit_button.pack(pady=10)

        # Play audio after the canvas is ready
        self.after(100, self.play_audio)  # Delay to ensure UI is loaded

    def play_audio(self):
        from pydub import AudioSegment
        from pydub.utils import which

        AudioSegment.converter = which("ffmpeg")  # Explicitly set ffmpeg path


        audio = AudioSegment.from_file('/home/ayden/Desktop/aydenprj/nathacks/dementions/Voice 001.m4a')
        data = np.array(audio.get_array_of_samples())

        # Ensure correct shape for stereo or mono
        if audio.channels == 2:
            data = data.reshape((-1, 2))

        # Play the audio
        sd.play(data, samplerate=audio.frame_rate)

        # Wait for the audio to finish
        sd.wait()

    def on_button_press(self, event):
        # Store the current mouse position
        self.last_x, self.last_y = event.x, event.y

    def on_mouse_drag(self, event):
        # Draw a line from the last position to the current position
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=2, fill="black", capstyle="round", smooth=True)
        self.last_x, self.last_y = event.x, event.y


    def submit_drawing(self):
        global predicted_class  # Update global predicted_class
        # In this example, we will simply show a message box confirming the drawing
        messagebox.showinfo("Drawing Submitted", "Your drawing has been submitted successfully!")
        
        # Optionally, you could save the canvas content to a file or reset the canvas
        self.save_drawing()

    def save_drawing(self):
        default_directory = os.path.expanduser("/home/ayden/Desktop/aydenprj/nathacks/dementions/")  # You can specify your desired directory
        file_name = "drawing.png"  # Default file name
        file_path = os.path.join(default_directory, file_name)

        # Save the canvas content as a PostScript file
        ps_filename = "temp.ps"
        self.canvas.postscript(file=ps_filename)

        # Open the PostScript file using Pillow and save as PNG
        img = Image.open(ps_filename)
        img.save(file_path, "PNG")

        classifier = ImageClassifier("/home/ayden/Desktop/aydenprj/nathacks/image_classifier.h5")

        try:
            predicted_class, confidence = classifier.classify_image("/home/ayden/Desktop/aydenprj/nathacks/dementions/drawing.png")
            predicted_class = int(predicted_class[-1])
            predicted_class = (100.0/7) *  predicted_class
            predicted_class = 100-predicted_class
            print(predicted_class)
        except FileNotFoundError:
            print("Image file not found. Please provide a valid file path.")
        
        self.destroy()  # Remove the DrawingApp frame completely
        # switch_frame(welcome_frame)
        
        hands_test()
        

def hands_test():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Create a new frame to attach the canvas
    hands_frame = ctk.CTkFrame(app)  # Use `app` or any valid parent frame
    hands_frame.grid(row=0, column=0, sticky="nsew")

    # Create the canvas within this frame
    hands_canvas = ctk.CTkCanvas(hands_frame, width=400, height=300)
    hands_canvas.pack(pady=10, padx=10)

    # Load and display the image
    image_path = "/home/ayden/Desktop/aydenprj/nathacks/dementions/IMG_2041.jpg"
    image = Image.open(image_path).resize((400, 300))
    photo = ImageTk.PhotoImage(image)

    # Display the image on the canvas
    hands_canvas.create_image(0, 0, anchor="nw", image=photo)
    hands_canvas.image = photo  # Keep a reference to avoid garbage collection

    # Allow the GUI to update before starting the camera
    hands_frame.update_idletasks()
    hands_canvas.create_text(
    200, 280,  # x, y coordinates (centered at the bottom)
    text="Please copy the hand position above",
    font=("Arial", 16, "bold"),
    fill="white")
    # Play audio before starting
    audio = AudioSegment.from_file('/home/ayden/Desktop/aydenprj/nathacks/dementions/Voice 002.m4a')
    data = np.array(audio.get_array_of_samples())

    if audio.channels == 2:
        data = data.reshape((-1, 2))

    sd.play(data, samplerate=audio.frame_rate)
    sd.wait()  # Wait for the audio to finish

    # Start camera processing after a delay
    app.after(1000, start_camera, mp_drawing, mp_hands)  # Delay of 1 second
    



def start_camera(mp_drawing, mp_hands):
    global stage, elapsed_time 
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Set the timeout duration (in seconds)
    timeout_duration = 40

    # Initialize Mediapipe hands
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        last_processing_time = time.time()
        processing_interval = 0.3  # Process a frame every 0.3 seconds
        start_time = time.time()  # Variable to store the time when fingers first touch
        elapsed_time = 0  # Track elapsed time
        end = False  # Track if fingers touched

        while True:
            success, image = cap.read()
            if not success:
                continue

            current_time = time.time()
            elapsed_time = current_time - start_time  # Update elapsed time

            # Exit if timeout is reached
            if elapsed_time >= timeout_duration:
                print("Timeout reached! Exiting...")
                break

            if current_time - last_processing_time >= processing_interval:
                last_processing_time = current_time

                # Convert the image to RGB for Mediapipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Access the landmarks for the thumb tip (landmark 4) and middle finger tip (landmark 12)
                        thumb_tip = hand_landmarks.landmark[4]
                        middle_tip = hand_landmarks.landmark[12]

                        # Get the coordinates of thumb and middle finger tips
                        thumb_tip_coords = (thumb_tip.x * image.shape[1], thumb_tip.y * image.shape[0])
                        middle_tip_coords = (middle_tip.x * image.shape[1], middle_tip.y * image.shape[0])

                        # Calculate the distance between thumb tip and middle finger tip
                        distance_thumb_middle = math.sqrt((thumb_tip_coords[0] - middle_tip_coords[0])**2 +
                                                        (thumb_tip_coords[1] - middle_tip_coords[1])**2)

                        # Define a threshold for when the fingers touch (distance <= 15 pixels)
                        if distance_thumb_middle < 15:
                            print(f"Fingers touched! Elapsed time: {elapsed_time:.2f} seconds")
                            end = True

                # Show the camera feed only if fingers haven't touched
                if not end:
                    cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(1) & 0xFF == 27 or end:  # Exit on 'Esc' or fingers touching
                break

        cap.release()
        cv2.destroyAllWindows()

    # Determine stage based on elapsed time
    stage = 0
    if elapsed_time < 4:
        stage = 1
    elif elapsed_time < 8:
        stage = 2
    elif elapsed_time < 15:
        stage = 3
    elif elapsed_time < 20:
        stage = 4
    elif elapsed_time < 20:
        stage = 5
    elif elapsed_time < 30:
        stage = 6
    else:
        stage = 7

    stage = stage*16

    print(f"Final Stage: {stage}")
    pygame_maze()
    
def pygame_maze():
    """Launches the Pygame Maze game."""
    for widget in app.winfo_children():
        widget.destroy()

    # Play audio before starting
    audio = AudioSegment.from_file('/home/ayden/Desktop/aydenprj/nathacks/dementions/Voice 003.m4a')
    data = np.array(audio.get_array_of_samples())

    if audio.channels == 2:
        data = data.reshape((-1, 2))

    sd.play(data, samplerate=audio.frame_rate)
    sd.wait()  # Wait for the audio to finish

    # Initialize pygame
    pygame.init()

    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maze with Timer")

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    # Maze layout
    MAZE = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    TILE_SIZE = WIDTH // len(MAZE[0])

    character = pygame.Rect(TILE_SIZE + 5, TILE_SIZE + 5, TILE_SIZE // 2, TILE_SIZE // 2)
    FINISH_POS = (TILE_SIZE * 14 + TILE_SIZE // 2, TILE_SIZE * 10 + TILE_SIZE // 2)
    finish_rect = pygame.Rect(FINISH_POS[0] - TILE_SIZE // 2, FINISH_POS[1] - TILE_SIZE // 2, TILE_SIZE, TILE_SIZE)

    start_time = time.time()
    time_limit = 200  # 5 minutes time limit

    def draw_maze():
        """Draw the maze and the start/finish points."""
        for row in range(len(MAZE)):
            for col in range(len(MAZE[0])):
                color = BLACK if MAZE[row][col] == 1 else WHITE
                pygame.draw.rect(screen, color, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        pygame.draw.rect(screen, GREEN, (TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE))  # Start
        pygame.draw.rect(screen, RED, finish_rect)  # Finish

    def follow_cursor():
        """Move character towards cursor while preventing wall collision."""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        dx, dy = mouse_x - character.centerx, mouse_y - character.centery
        distance = (dx**2 + dy**2) ** 0.5
        if distance > 0:
            dx, dy = dx / distance * 5, dy / distance * 5

        new_rect = character.move(dx, dy)
        for row in range(len(MAZE)):
            for col in range(len(MAZE[0])):
                if MAZE[row][col] == 1 and new_rect.colliderect(pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)):
                    return  # Collision, stop movement

        character.move_ip(dx, dy)

    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(WHITE)
        draw_maze()
        pygame.draw.ellipse(screen, BLUE, character)

        follow_cursor()

        elapsed_time = time.time() - start_time
        if character.colliderect(finish_rect):
            pygame.time.wait(2000)  # Show completion
            break
        elif elapsed_time > time_limit:
            print("Time's up!")
            pygame.time.wait(2000)
            elapsed_time = 200
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    elasped_time = elapsed_time/2
    print(elapsed_time)
    eeg()
    # time.sleep(10)
    # pass_to_admin()


def eeg():
    """Displays a message, plays audio, and shows a calming GIF for the EEG task."""
    # Clear existing widgets
    for widget in app.winfo_children():
        widget.destroy()

    audio = AudioSegment.from_file('/home/ayden/Desktop/aydenprj/nathacks/dementions/Voice 004.m4a')
    data = np.array(audio.get_array_of_samples())

    if audio.channels == 2:
        data = data.reshape((-1, 2))

    sd.play(data, samplerate=audio.frame_rate)
    sd.wait()  # Wait for the audio to finish

    gif_path = "dementions/Screen-Recording-ezgif.com-video-to-gif-converter.gif"
    parent_frame = ctk.CTkFrame(app)
    parent_frame.pack(fill="both", expand=True, padx=20, pady=20)

    try:
        if not os.path.exists(gif_path):
            raise FileNotFoundError(f"GIF file not found: {gif_path}")

        img = Image.open(gif_path)
        frames = []

        try:
            while True:
                frames.append(ImageTk.PhotoImage(img.copy()))
                img.seek(len(frames))  # Load next frame
        except EOFError:
            pass

        if not frames:
            raise ValueError("No frames loaded from GIF.")

        gif_label = ctk.CTkLabel(parent_frame, text="")
        gif_label.pack(pady=20)

        def animate(index):
            if not parent_frame.winfo_exists():
                return  # Stop animation if frame is destroyed

            gif_label.configure(image=frames[index])
            parent_frame.update_idletasks()
            next_index = (index + 1) % len(frames)
            parent_frame.after(100, animate, next_index)

        def stop_animation():
            gif_label.pack_forget()
            proceed_to_next_task()  # Call the next task here

        def proceed_to_next_task():
            pass_to_admin()
        animate(0)  # Start the animation immediately
        parent_frame.after(20000, stop_animation)  # Stop after 20 seconds

    except Exception as e:
        print(f"Error displaying GIF: {e}")


    
 


def check_password(password_entry):
    password = password_entry.get()
    if password == "1234":
        admin_form()
    else:
        messagebox.showerror("Access Denied", "Incorrect Password. Please try again.")


def admin_form():
    """Displays a compact form with 10 questions for the caretaker to answer."""
    # Clear all widgets from the app
    for widget in app.winfo_children():
        widget.destroy()

    # Create a frame for the questions
    form_frame = ctk.CTkFrame(app)
    form_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # List of 10 questions where 10 represents a negative outcome
    questions = [
        "How frequently does the patient lose focus?",
        "How frequently does the patient forget recent events?",
        "How frequently does the patient require assistance?",
        "How frequently does the patient poorly manage daily tasks?",
        "How frequently is the patient in a bad mood?",
        "How physically inactive is the patient?",
        "How often does the patient struggle to communicate?",
        "How often does the patient experience poor sleep quality?",
        "How often does the patient struggle to follow instructions?",
        "How frequently does the patient express confusion?"
    ]

    # Store responses
    responses = []

    # Create labels and sliders for each question
    for i, question in enumerate(questions, start=1):
        # Question Label
        question_label = ctk.CTkLabel(
            form_frame, text=f"{i}. {question}", font=("Arial", 12, "bold"), anchor="w"
        )
        question_label.pack(fill="x", pady=(3, 2))

        # Frame for slider and its side labels
        slider_frame = ctk.CTkFrame(form_frame, height=40)
        slider_frame.pack(fill="x", pady=1)

        # Left side label for positive
        left_label = ctk.CTkLabel(slider_frame, text="1 - Rarely", font=("Arial", 8))
        left_label.grid(row=0, column=0, padx=3, sticky="w")

        # Slider for response
        slider = ctk.CTkSlider(slider_frame, from_=1, to=10, number_of_steps=9, height=8)
        slider.set(5)  # Default value
        slider.grid(row=0, column=1, padx=8, sticky="ew")

        # Add small tick mark for the middle
        tick_label = ctk.CTkLabel(slider_frame, text="5", font=("Arial", 6))
        tick_label.grid(row=1, column=1, sticky="n")
        slider_frame.grid_columnconfigure(1, weight=1)

        # Right side label for negative
        right_label = ctk.CTkLabel(slider_frame, text="10 - Frequently", font=("Arial", 8))
        right_label.grid(row=0, column=2, padx=3, sticky="e")

        responses.append(slider)



    # Submit button
    def submit_form():
        global results_fit
        results = [slider.get() for slider in responses]
        print(results)
        results_fit = sum(results)
        print(sum(results))
        results_display()

    submit_button = ctk.CTkButton(app, text="Submit", command=submit_form)
    submit_button.pack(pady=20)

        
def pass_to_admin():
    # Clear all widgets from the app
    for widget in app.winfo_children():
        widget.destroy()

    # Create a canvas for text display
    canvas = ctk.CTkCanvas(app, bg="white", width=400, height=300)
    canvas.pack(pady=20)

    # Add "Pass to Caretaker" text
    canvas.create_text(
        200, 50,  # Centered position on canvas
        text="Pass to Caretaker",
        font=("Arial", 20, "bold"),
        fill="black"
    )

    # Create password input field
    password_entry = ctk.CTkEntry(app, placeholder_text="Enter Password", show="*")
    password_entry.pack(pady=10)

    # Submit button with lambda to pass password_entry
    password_submit_button = ctk.CTkButton(
        app, 
        text="Submit", 
        command=lambda: check_password(password_entry)
    )
    password_submit_button.pack(pady=10)

def results_display():
    """Displays the calculated results for physical, behavior, speech, emotion, and memory on the app."""
    print(predicted_class, stage, results_fit, speech_score, emotion_score, elapsed_time)
    
    for widget in app.winfo_children():
        widget.destroy()
    
    # Calculate metrics
    physical = (predicted_class + stage) / 2
    behavior = results_fit
    speech = speech_score
    emotion = emotion_score
    memory = elapsed_time
    
    # Create a frame for the results
    results_frame = ctk.CTkFrame(app)
    results_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Title Label
    title_label = ctk.CTkLabel(results_frame, text="Results Summary", font=("Arial", 18, "bold"))
    title_label.pack(pady=10)

    # Display each result
    result_labels = [
        f"Physical: {physical:.2f}",
        f"Behavior: {behavior}",
        f"Speech: {speech}",
        f"Emotion: {emotion}",
        f"Memory: {memory}"
    ]

    for result in result_labels:
        label = ctk.CTkLabel(results_frame, text=result, font=("Arial", 14))
        label.pack(pady=5)
    
    # Next Step Button
    next_button = ctk.CTkButton(results_frame, text="Next Task", command=next_task)
    next_button.pack(pady=20)

def next_task():
    """Placeholder function for the next task."""
    print("Proceeding to the next task...")




def sign_in():
    name = username_entry.get()
    if name.strip():  # Check if the name is non-empty
        clear_window()  # Clear everything
    else:
        messagebox.showerror("Error", "Please enter your name")

# Sign-up logic
def sign_up():
    name = name_entry.get()
    age = age_entry.get()
    gender = gender_entry.get()
    race = race_entry.get()
    address = address_entry.get()

    if all([name.strip(), age.strip(), gender.strip(), race.strip(), address.strip()]):
        clear_window()  # Clear everything
        messagebox.showinfo("Success", f"Account created for {name}!")
    else:
        messagebox.showerror("Error", "Please fill all fields")

# Frames: Ensure they expand to fill the window
welcome_frame = ctk.CTkFrame(app)
signin_frame = ctk.CTkFrame(app)
signup_frame = ctk.CTkFrame(app)

for frame in (welcome_frame, signin_frame, signup_frame):
    frame.grid(row=0, column=0, sticky="nsew")

# Welcome Page
welcome_label = ctk.CTkLabel(welcome_frame, text="Welcome to the App!", font=("Arial", 24))
welcome_label.pack(pady=20)
signin_button = ctk.CTkButton(welcome_frame, text="Sign In", command=lambda: switch_frame(signin_frame), width=200)
signin_button.pack(pady=10)
signup_button = ctk.CTkButton(welcome_frame, text="Sign Up", command=lambda: switch_frame(signup_frame), width=200)
signup_button.pack(pady=10)

# Sign In Page
signin_label = ctk.CTkLabel(signin_frame, text="Sign In", font=("Arial", 24))
signin_label.pack(pady=20, padx=20)
username_label = ctk.CTkLabel(signin_frame, text="Name:")
username_label.pack(pady=5)
username_entry = ctk.CTkEntry(signin_frame, width=300)
username_entry.pack(pady=5)
signin_submit_button = ctk.CTkButton(signin_frame, text="Submit", command=sign_in, width=200)
signin_submit_button.pack(pady=10)
back_button1 = ctk.CTkButton(signin_frame, text="Back", command=lambda: switch_frame(welcome_frame), width=200)
back_button1.pack(pady=10)

# Sign Up Page
signup_label = ctk.CTkLabel(signup_frame, text="Sign Up", font=("Arial", 36))
signup_label.grid(row=0, column=0, pady=30, sticky="n")

# Configure Sign Up Page elements to align centrally
form_frame = ctk.CTkFrame(signup_frame)
form_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
signup_frame.grid_rowconfigure(1, weight=1)  # Allow form frame to expand

form_frame.grid_columnconfigure(0, weight=1)
form_frame.grid_columnconfigure(1, weight=2)

name_label = ctk.CTkLabel(form_frame, text="Name:")
name_label.grid(row=0, column=0, pady=10, padx=10, sticky="e")
name_entry = ctk.CTkEntry(form_frame)
name_entry.grid(row=0, column=1, pady=10, padx=10, sticky="ew")

age_label = ctk.CTkLabel(form_frame, text="Age:")
age_label.grid(row=1, column=0, pady=10, padx=10, sticky="e")
age_entry = ctk.CTkEntry(form_frame)
age_entry.grid(row=1, column=1, pady=10, padx=10, sticky="ew")

gender_label = ctk.CTkLabel(form_frame, text="Gender:")
gender_label.grid(row=2, column=0, pady=10, padx=10, sticky="e")
gender_entry = ctk.CTkEntry(form_frame)
gender_entry.grid(row=2, column=1, pady=10, padx=10, sticky="ew")

race_label = ctk.CTkLabel(form_frame, text="Race:")
race_label.grid(row=3, column=0, pady=10, padx=10, sticky="e")
race_entry = ctk.CTkEntry(form_frame)
race_entry.grid(row=3, column=1, pady=10, padx=10, sticky="ew")

address_label = ctk.CTkLabel(form_frame, text="Address:")
address_label.grid(row=4, column=0, pady=10, padx=10, sticky="e")
address_entry = ctk.CTkEntry(form_frame)
address_entry.grid(row=4, column=1, pady=10, padx=10, sticky="ew")

# Buttons Section
button_frame = ctk.CTkFrame(signup_frame)
button_frame.grid(row=2, column=0, pady=30, sticky="n")
signup_submit_button = ctk.CTkButton(button_frame, text="Register", command=sign_up, width=200)
signup_submit_button.pack(pady=10)
back_button2 = ctk.CTkButton(button_frame, text="Back", command=lambda: switch_frame(welcome_frame), width=200)
back_button2.pack(pady=10)

# Show the welcome page initially
switch_frame(welcome_frame)

# Run the app
app.mainloop()
