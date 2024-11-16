import cv2
import math
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    last_processing_time = time.time()
    processing_interval = 0.3  # Process a frame every 0.3 seconds
    start_time = time.time()  # Variable to store the time when fingers first touch

    while True:
        success, image = cap.read()
        if not success:
            continue
        
        current_time = time.time()
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
                        elapsed_time = time.time() - start_time
                        print(f"Fingers touched! Elapsed time: {elapsed_time:.2f} seconds")
                        
                        # Stop the camera feed and show a black frame
                        black_frame = np.zeros_like(image)  # Create a black frame of the same size as the input image
                        cv2.imshow('MediaPipe Hands', black_frame)  # Display black frame
                        cap.release()
                        cv2.destroyAllWindows()
                        break  # Stop the program

            # Only show the camera feed if no hand is detected or fingers are not touching
            cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
