import pandas as pd
import muselsl

def modify_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

#Drop the first column
    df.drop(df.columns[0], axis=1, inplace=True)

#Move the "timestamp" column to the first position
    timestamp_col = df.pop("timestamps")
    df.insert(0, "timestamp", timestamp_col)

#Save the modified CSV
    df.to_csv(output_file, index=False)
    print(f"Modified CSV saved as {output_file}")

muselsl.record(30, filename='eeg-feature-generation-master\dataset\MUSE2/muse_recording.csv')

# Optionally modify CSV output
# modify_csv(input_csv, output_csv)