
# Start the streaming script in a new process
def run():
    import subprocess
    import time
    import predictions
    import os
    import EEG_generate_training_matrix
    import EEG_feature_extraction
    streaming_process = subprocess.Popen(['python', 'eeg-feature-generation-master\code\stream.py'])

    # Wait for the stream to start (adjust time if necessary)
    time.sleep(10)

    # Start the recording script in another process
    recording_process = subprocess.Popen(['python', 'eeg-feature-generation-master\code/recordings.py'])

    # Wait for the recording process to finish
    recording_process.wait()

    # Optionally, terminate the streaming process after recording
    streaming_process.terminate()

    emotions = predictions.predict_emotion('eeg-feature-generation-master\dataset\MUSE2')
    os.remove('eeg-feature-generation-master\dataset\MUSE2\muse_recording.csv')

    def detect_mood_swings(sentiment_array):
        """
        Detects mood swings and calculates a normalized score (0-100) based on the violence and frequency of swings.

        Parameters:
        sentiment_array (list): A list of sentiment strings ('positive', 'negative', 'neutral').

        Returns:
        dict: {
            'swings': list of tuples indicating mood swing details,
            'score': normalized score (0-100),
        }
        """
        streaks = []  # List to store streaks and their indices
        mood_swings = []  # List to store detected mood swings
        
        # Step 1: Identify streaks
        current_streak = [0, sentiment_array[0]]  # [start_index, sentiment]
        for i in range(1, len(sentiment_array)):
            if sentiment_array[i] == current_streak[1]:  # Continue the streak
                continue
            else:  # Streak ended
                streaks.append((current_streak[0], i - 1, current_streak[1]))
                current_streak = [i, sentiment_array[i]]  # Start a new streak
        # Append the last streak
        streaks.append((current_streak[0], len(sentiment_array) - 1, current_streak[1]))
        
        # Step 2: Detect mood swings between streaks
        for i in range(1, len(streaks)):
            prev_streak = streaks[i - 1]
            current_streak = streaks[i]
            # Detect all transitions: positive <-> negative, and neutral <-> positive/negative
            if (prev_streak[2] == "positive" and current_streak[2] == "negative") or \
            (prev_streak[2] == "negative" and current_streak[2] == "positive"):
                severity = 2  # Strong transition
            elif (prev_streak[2] == "neutral" and current_streak[2] in ["positive", "negative"]) or \
                (current_streak[2] == "neutral" and prev_streak[2] in ["positive", "negative"]):
                severity = 0.5  # Moderate transition
            else:
                continue  # Ignore transitions like neutral -> neutral
            
            mood_swings.append({
                'from': prev_streak[2],
                'to': current_streak[2],
                'start_index': prev_streak[1],
                'end_index': current_streak[0],
                'prev_length': prev_streak[1] - prev_streak[0] + 1,
                'current_length': current_streak[1] - current_streak[0] + 1,
                'severity': severity
            })
        
        # Step 3: Calculate score based on swing severity and frequency
        total_swings = len(mood_swings)
        total_streak_length = sum(swing['prev_length'] + swing['current_length'] for swing in mood_swings)
        total_severity = sum(swing['severity'] * abs(swing['prev_length'] - swing['current_length']) for swing in mood_swings)
        
        # Scoring formula: Combine frequency, streak lengths, and severity
        if total_swings == 0:
            score = 0
        else:
            # Normalize components (weights can be adjusted for sensitivity)
            normalized_swings = min(total_swings / len(sentiment_array) * 75, 100)
            normalized_length = min(total_streak_length / len(sentiment_array) * 75, 100)
            normalized_severity = min(total_severity / (len(sentiment_array) * 2) * 75, 100)
            
            # Final score combines all three factors with equal weight
            score = (normalized_swings + normalized_length + normalized_severity) / 3
        
        return round(score, 2)

    score = detect_mood_swings(emotions)
    print(score)
    return score