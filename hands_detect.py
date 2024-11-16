import cv2
import math
import mediapipe as mp
import time

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

print(f"Final Stage: {stage}")
