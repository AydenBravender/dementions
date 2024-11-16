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

    # Create new widgets for drawing interface
    drawing_app = DrawingApp(app)
    drawing_app.pack(fill="both", expand=True)


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

    def on_button_press(self, event):
        # Store the current mouse position
        self.last_x, self.last_y = event.x, event.y

    def on_mouse_drag(self, event):
        # Draw a line from the last position to the current position
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=2, fill="black", capstyle="round", smooth=True)
        self.last_x, self.last_y = event.x, event.y

    def submit_drawing(self):
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

    # Start camera processing after a delay
    app.after(1000, start_camera, mp_drawing, mp_hands)  # Delay of 1 second
    



def start_camera(mp_drawing, mp_hands):
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
    pass_to_admin()

def check_password(password_entry):
    password = password_entry.get()
    if password == "1234":
        admin_form()
    else:
        messagebox.showerror("Access Denied", "Incorrect Password. Please try again.")


import customtkinter as ctk

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
        results = [slider.get() for slider in responses]
        messagebox.showinfo("Form Submitted", f"Thank you! Responses: {results}")
        pass_to_admin()

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

def sign_in():
    name = username_entry.get()
    if name.strip():  # Check if the name is non-empty
        clear_window()  # Clear everything
        messagebox.showinfo("Success", f"Welcome, {name}!")
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
