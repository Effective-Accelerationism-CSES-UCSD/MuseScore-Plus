import pickle
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

def plot_piano_roll(notes_df, title="Midi Viewer"):
    plt.figure(figsize=(12, 6))
    
    if notes_df.empty:
        print(f"No notes found in {title}")
        return
    
    for _, row in notes_df.iterrows():
        plt.barh(row["Pitch"], row["Duration"], left=row["StartTime"], height=0.5, color="blue")
    
    plt.xlabel("Time (Absolute)")
    plt.ylabel("Pitch")
    plt.title(title)
    #plt.gca().invert_yaxis()
    plt.show()

def load_pickle(pkl_file):
    if not os.path.isfile(pkl_file) or not pkl_file.endswith(".pkl"):
        print("Provide a valid .pkl file.")
        sys.exit(1)

    with open(pkl_file, "rb") as f:
        return pickle.load(f)

def visualize_pickle(pkl_file):
    data = load_pickle(pkl_file)

    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        for filename, midi_info in data.items():
            if "notes" in midi_info and isinstance(midi_info["notes"], pd.DataFrame):
                print(f"Visualizing: {filename}")
                plot_piano_roll(midi_info["notes"], title=f"Midi View - {filename}")
            else:
                print(f"Warning: Invalid file {filename}")

    elif isinstance(data, dict) and "notes" in data and isinstance(data["notes"], pd.DataFrame):
        plot_piano_roll(data["notes"], title="Midi Viewer")

    else:
        print("Invalid pickle structure. Must have 'notes' and 'channels' keys.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python midi_viewer.py <pickle_file>")
        sys.exit(1)

    visualize_pickle(sys.argv[1])
