import torch
import torch.nn as nn
import pickle
import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from collections import defaultdict
import sys

from transformer_consistency import CrossAttentionTransformer, normalize_features, calculate_consistency_score

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path, input_dim=4, model_dim=128, num_heads=16, num_layers=8, num_classes=16):
    model = CrossAttentionTransformer(input_dim, model_dim, num_heads, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    print(f"Loaded CrossAttentionTransformer model from {model_path}")
    return model

def load_single_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        df, channels = pickle.load(f)
    
    features = df[["Pitch", "Velocity", "StartTime", "Duration"]].values
    features_tensor = torch.tensor(features, dtype=torch.float)
    
    features_tensor = normalize_features(features_tensor)
    
    targets_tensor = torch.tensor(channels, dtype=torch.long)
    
    return features_tensor, targets_tensor, df

def evaluate_clustering_consistency(true_labels, pred_labels):
    rand_score = adjusted_rand_score(true_labels, pred_labels)
    ami_score = adjusted_mutual_info_score(true_labels, pred_labels)

    consistency_score = calculate_consistency_score(
        torch.tensor(true_labels), 
        torch.tensor(pred_labels)
    )
    
    # Create label mapping for analysis
    label_mapping = {}
    for pred_label in np.unique(pred_labels):
        mask = (pred_labels == pred_label)
        true_labels_in_cluster = true_labels[mask]
        
        if len(true_labels_in_cluster) == 0:
            continue
            
        unique_true, counts = np.unique(true_labels_in_cluster, return_counts=True)
        most_common_true = unique_true[np.argmax(counts)]
        label_mapping[pred_label] = most_common_true
    
    return rand_score, ami_score, label_mapping, consistency_score

def test_model_consistency(model, pkl_path, num_examples=20, batch_size=128):
    features, targets, df = load_single_pkl(pkl_path)
    total = len(targets)
    
    all_predictions = []
    
    for i in range(0, total, batch_size):
        batch_features = features[i:i+batch_size].to(device)
        
        with torch.no_grad():
            logits = model(batch_features)
            batch_predictions = torch.argmax(logits, dim=-1)
            all_predictions.append(batch_predictions.cpu())
    
    predictions = torch.cat(all_predictions).numpy()
    targets_np = targets.numpy()

    rand_score, ami_score, label_mapping, consistency_score = evaluate_clustering_consistency(
        targets_np, predictions
    )

    print(f"\n===== Model Clustering Evaluation =====")
    print(f"Test file: {pkl_path}")
    print(f"Total Notes: {total}")
    print(f"Adjusted Rand Index: {rand_score:.4f}")
    print(f"Adjusted Mutual Information: {ami_score:.4f}")
    print(f"Consistency Score: {consistency_score:.4f}")

    print("\n===== Cluster Analysis =====")
    true_clusters = defaultdict(int)
    pred_clusters = defaultdict(int)
    
    for true, pred in zip(targets_np, predictions):
        true_clusters[true] += 1
        pred_clusters[pred] += 1
    
    print(f"Number of true clusters: {len(true_clusters)}")
    print(f"Number of predicted clusters: {len(pred_clusters)}")

    print("\nTop 5 true clusters by size:")
    for cluster, count in sorted(true_clusters.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Channel {cluster}: {count} notes ({count/total*100:.2f}%)")
    
    print("\nTop 5 predicted clusters by size:")
    for cluster, count in sorted(pred_clusters.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Predicted {cluster}: {count} notes ({count/total*100:.2f}%)")

    print("\n===== Cluster Mapping =====")
    print("Predicted -> Most Common True Channel")
    for pred, true in sorted(label_mapping.items()):
        pred_count = pred_clusters[pred]
        print(f"  Predicted {pred} -> Channel {true} ({pred_count} notes, {pred_count/total*100:.2f}%)")

    print("\n===== Random Sample Clustering =====")
    sample_indices = random.sample(range(total), min(num_examples, total))

    pitches = df["Pitch"].values
    start_times = df["StartTime"].values
    
    for i in sample_indices:
        true_label = targets_np[i]
        pred_label = predictions[i]
        mapped_label = label_mapping.get(pred_label, "Unknown")
        consistent = (mapped_label == true_label)
        
        print(f"Note {i}: Pitch={pitches[i]:.1f}, Time={start_times[i]:.2f}, " +
              f"True Channel={true_label}, Predicted={pred_label}, " +
              f"Consistency={'✓' if consistent else '✗'}")

    print("\n===== Channel Consistency =====")
    channel_consistency = {}
    
    for true_channel in sorted(true_clusters.keys()):
        mask = (targets_np == true_channel)
        preds_for_channel = predictions[mask]
        
        # Find the most common predicted label for true channel
        unique_preds, counts = np.unique(preds_for_channel, return_counts=True)
        most_common_pred = unique_preds[np.argmax(counts)]
        correct_count = np.sum(preds_for_channel == most_common_pred)
        total_count = len(preds_for_channel)
        
        channel_consistency[true_channel] = correct_count / total_count

    for channel, consistency in sorted(channel_consistency.items(), key=lambda x: true_clusters[x[0]], reverse=True)[:5]:
        count = true_clusters[channel]
        print(f"  Channel {channel} ({count} notes): {consistency:.4f} consistency")

    df["PredictedChannel"] = predictions
    
    return rand_score, ami_score, consistency_score, predictions, df

def analyze_feature_impact_on_consistency(model, pkl_path, num_samples=500):
    features, targets, _ = load_single_pkl(pkl_path)
    
    if num_samples < len(features):
        indices = random.sample(range(len(features)), num_samples)
        features = features[indices]
        targets = targets[indices]
    
    features = features.to(device)
    targets = targets.numpy()
    
    model.eval()
    with torch.no_grad():
        baseline_logits = model(features)
        baseline_preds = torch.argmax(baseline_logits, dim=-1).cpu().numpy()
    
    _, _, _, baseline_consistency = evaluate_clustering_consistency(targets, baseline_preds)
    
    feature_names = ["Pitch", "Velocity", "StartTime", "Duration"]
    impact_scores = []
    
    for i in range(4):
        # Make a copy of features and perturb one dimension
        perturbed = features.clone()
        perturbed[:, i] = torch.randn_like(perturbed[:, i]) * perturbed[:, i].std() + perturbed[:, i].mean()
        
        # Get predictions with perturbed features
        with torch.no_grad():
            perturbed_logits = model(perturbed)
            perturbed_preds = torch.argmax(perturbed_logits, dim=-1).cpu().numpy()
        
        # Calculate consistency with perturbed feature
        _, _, _, perturbed_consistency = evaluate_clustering_consistency(targets, perturbed_preds)
        
        # Calculate impact as reduction in consistency
        impact = baseline_consistency - perturbed_consistency
        impact_scores.append(impact)
    
    print("\n===== Feature Impact on Clustering Consistency =====")
    print(f"Baseline consistency: {baseline_consistency:.4f}")
    for name, score in zip(feature_names, impact_scores):
        print(f"{name}: {score:.4f} impact (higher means more important)")
    
    return dict(zip(feature_names, impact_scores))

def get_color_map(labels):
    """Generate a colormap for the unique labels"""
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    cmap = plt.cm.get_cmap('tab20', num_classes)
    colors = {label: cmap(i) for i, label in enumerate(unique_labels)}
    
    return colors

def plot_piano_roll(df, channel_column, title="MIDI Piano Roll", figsize=(15, 8), show_legend=True):
    plt.figure(figsize=figsize)

    channels = df[channel_column].values
    color_map = get_color_map(channels)

    for _, row in df.iterrows():
        channel = row[channel_column]
        color = color_map[channel]
        plt.barh(
            row["Pitch"], 
            row["Duration"], 
            left=row["StartTime"], 
            height=0.8, 
            color=color, 
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
    
    if show_legend:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[channel], label=f'Channel {channel}')
                           for channel in sorted(color_map.keys())]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pitch (MIDI note number)")
    plt.title(title)

    min_pitch = max(0, df["Pitch"].min() - 5)
    max_pitch = min(127, df["Pitch"].max() + 5)
    plt.ylim(min_pitch, max_pitch)
    
    # Add grid
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_piano_rolls(df, true_col="Channel", pred_col="PredictedChannel", figsize=(15, 10)):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    axes[0].set_title(f"True Channels ({true_col})")
    true_colors = get_color_map(df[true_col].values)
    
    axes[1].set_title(f"Predicted Channels ({pred_col})")
    pred_colors = get_color_map(df[pred_col].values)
    
    for _, row in df.iterrows():
        true_channel = row[true_col]
        pred_channel = row[pred_col]
        
        # True channels (top)
        axes[0].barh(
            row["Pitch"], 
            row["Duration"], 
            left=row["StartTime"], 
            height=0.8, 
            color=true_colors[true_channel], 
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Predicted channels (bottom)
        axes[1].barh(
            row["Pitch"], 
            row["Duration"], 
            left=row["StartTime"], 
            height=0.8, 
            color=pred_colors[pred_channel], 
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
    
    from matplotlib.patches import Patch
    true_legend = [Patch(facecolor=true_colors[ch], label=f'Channel {ch}') 
                  for ch in sorted(true_colors.keys())]
    axes[0].legend(handles=true_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    pred_legend = [Patch(facecolor=pred_colors[ch], label=f'Channel {ch}') 
                  for ch in sorted(pred_colors.keys())]
    axes[1].legend(handles=pred_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for ax in axes:
        ax.set_ylabel("Pitch (MIDI note number)")
        ax.grid(alpha=0.3)
    
    axes[1].set_xlabel("Time (seconds)")

    min_pitch = max(0, df["Pitch"].min() - 5)
    max_pitch = min(127, df["Pitch"].max() + 5)
    for ax in axes:
        ax.set_ylim(min_pitch, max_pitch)
    
    plt.tight_layout()
    plt.show()

def interactive_midi_viewer(df):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Button
    
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(left=0.3)
    
    current_view = 'True Channels'
    
    def draw():
        ax.clear()
        if current_view == 'True Channels':
            title = "MIDI Piano Roll - True Channels"
            channel_col = "Channel"
        else:
            title = "MIDI Piano Roll - Predicted Channels"
            channel_col = "PredictedChannel"
        
        # Get color map for the selected channel
        channels = df[channel_col].values
        color_map = get_color_map(channels)
        
        # Plot each note
        for _, row in df.iterrows():
            channel = row[channel_col]
            color = color_map[channel]
            ax.barh(
                row["Pitch"], 
                row["Duration"], 
                left=row["StartTime"], 
                height=0.8, 
                color=color, 
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )
        

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[ch], label=f'Channel {ch}') 
                          for ch in sorted(color_map.keys())]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Pitch (MIDI note number)")
        
        # Set reasonable y-axis limits
        min_pitch = max(0, df["Pitch"].min() - 5)
        max_pitch = min(127, df["Pitch"].max() + 5)
        ax.set_ylim(min_pitch, max_pitch)
        
        ax.grid(alpha=0.3)
        
        plt.draw()
    
    # Radio button callback
    def change_view(label):
        nonlocal current_view
        current_view = label
        draw()
    
    # Create radio buttons
    ax_radio = plt.axes([0.05, 0.7, 0.15, 0.15])
    radio = RadioButtons(ax_radio, ('True Channels', 'Predicted Channels'))
    radio.on_clicked(change_view)
    
    # Create a button to toggle comparison view
    ax_button = plt.axes([0.05, 0.5, 0.15, 0.1])
    button = Button(ax_button, 'Show Comparison')
    
    def show_comparison(event):
        plt.close(fig)
        compare_piano_rolls(df)
    
    button.on_clicked(show_comparison)
    
    # Initial draw
    draw()
    
    plt.show()

def visualize_midi(pkl_file, model_path):
    """
    Load a MIDI pickle file, run the model on it, and visualize the results
    """
    # Load the model
    model = load_model(model_path)
    
    # Get the ground truth data first
    features, targets, original_df = load_single_pkl(pkl_file)
    
    # Run the model on the pkl file
    _, _, _, predictions, df_with_predictions = test_model_consistency(model, pkl_file)
    
    # Add the true channel data to the DataFrame
    if "Channel" not in df_with_predictions.columns:
        df_with_predictions["Channel"] = targets.numpy()
    
    # Now launch the interactive viewer
    interactive_midi_viewer(df_with_predictions)
    
    # Also show the side-by-side comparison
    compare_piano_rolls(df_with_predictions)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <pickle_file> [model_path]")
        print("  pickle_file: Path to the MIDI pickle file to visualize")
        print("  model_path: Optional path to the model file (default: midi_transformer_consistency.pth)")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "midi_transformer_consistency-90.pth"
    
    visualize_midi(pkl_file, model_path)