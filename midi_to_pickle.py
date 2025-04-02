import mido
from mido import MidiFile
import pandas as pd
import pickle
import os
import sys

# Function to extract notes from MIDI file
def extract_midi_notes(file_path):
    midi = MidiFile(file_path)
    active_notes = {}  # Dict to track note_on events { (channel, pitch) -> (velocity, start_time) }
    notes = []  # List of extracted notes
    channels = []  # Separate list for channel tracking
    absolute_time = 0  # Running absolute time
    
    for track in midi.tracks:
        absolute_time = 0  # Reset absolute time per track
        for msg in track:
            absolute_time += msg.time  # Convert relative time to absolute time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[(msg.channel, msg.note)] = (msg.velocity, absolute_time)
                
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and (msg.channel, msg.note) in active_notes:
                velocity, start_time = active_notes.pop((msg.channel, msg.note))
                duration = absolute_time - start_time
                notes.append((msg.note, velocity, start_time, duration))
                channels.append(msg.channel)  # Store channel separately
    
    return notes, channels

# Process the input
def process_input(input_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output_midi")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(input_path) and input_path.endswith(".mid"):
        process_file(input_path, output_dir)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for filename in files:
                if filename.endswith(".mid"):
                    file_path = os.path.join(root, filename)
                    process_file(file_path, output_dir)
    else:
        print("Provide a valid MIDI file or directory.")
        sys.exit(1)

# Process a file
def process_file(input_path, output_dir):
    try:
        notes, channels = extract_midi_notes(input_path)
        df = pd.DataFrame(notes, columns=["Pitch", "Velocity", "StartTime", "Duration"])
        channel_series = pd.Series(channels)
        
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + ".pkl")
        with open(output_file, "wb") as f:
            pd.to_pickle((df, channel_series), f)

        print(f"Processed: {input_path}")

    except Exception as e:
        print(f"Skipping {input_path} due to error: {e}")

# Run script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <midi_file_or_directory>")
        sys.exit(1)
    process_input(sys.argv[1])
