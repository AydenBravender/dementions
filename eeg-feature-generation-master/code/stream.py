# stream.py
import muselsl
import sys

# List available Muse devices
muses = muselsl.list_muses()
print(muses)

# Start streaming from the first Muse device
muselsl.stream(muses[0]['address'])
print('e')

# Keep the stream active (or let it stream indefinitely until interrupted)
print("Streaming started... Press Ctrl+C to stop.")
try:
    while True:
        pass  # Keeps the script running
except KeyboardInterrupt:
    print("Streaming stopped.")
    sys.exit()