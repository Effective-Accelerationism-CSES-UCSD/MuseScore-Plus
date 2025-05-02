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
import glob

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

def test_model_consistency(model, pkl_path, num_examples=20, batch_size=128, verbose=True):
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

    if verbose:
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

def process_directory(model, directory_path, batch_size=128, file_pattern="*.pkl"):
    """
    Process all pickle files in a directory and calculate average consistency scores
    """
    # Find all pickle files in the directory
    pkl_files = glob.glob(os.path.join(directory_path, file_pattern))
    
    if not pkl_files:
        print(f"No pickle files found in {directory_path} matching pattern {file_pattern}")
        return None
    
    print(f"Found {len(pkl_files)} pickle files in {directory_path}")
    
    # Initialize score accumulators
    total_rand_score = 0.0
    total_ami_score = 0.0
    total_consistency_score = 0.0
    file_results = []
    
    # Process each file
    for i, pkl_file in enumerate(pkl_files):
        print(f"\nProcessing file {i+1}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
        try:
            rand_score, ami_score, consistency_score, _, _ = test_model_consistency(
                model, pkl_file, batch_size=batch_size, verbose=False
            )
            
            # Add to totals
            total_rand_score += rand_score
            total_ami_score += ami_score
            total_consistency_score += consistency_score
            
            # Store individual file results
            file_results.append({
                'file': os.path.basename(pkl_file),
                'rand_score': rand_score,
                'ami_score': ami_score,
                'consistency_score': consistency_score
            })
            
            print(f"  Adjusted Rand Index: {rand_score:.4f}")
            print(f"  Adjusted Mutual Information: {ami_score:.4f}")
            print(f"  Consistency Score: {consistency_score:.4f}")
            
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    
    # Calculate averages
    num_processed = len(file_results)
    if num_processed == 0:
        print("No files were successfully processed.")
        return None
    
    avg_rand_score = total_rand_score / num_processed
    avg_ami_score = total_ami_score / num_processed
    avg_consistency_score = total_consistency_score / num_processed
    
    # Print summary
    print("\n===== Directory Analysis Summary =====")
    print(f"Processed {num_processed} files")
    print(f"Average Adjusted Rand Index: {avg_rand_score:.4f}")
    print(f"Average Adjusted Mutual Information: {avg_ami_score:.4f}")
    print(f"Average Consistency Score: {avg_consistency_score:.4f}")
    
    # Sort files by consistency score
    file_results.sort(key=lambda x: x['consistency_score'], reverse=True)
    
    print("\n===== Files Ranked by Consistency Score =====")
    for i, result in enumerate(file_results):
        print(f"{i+1}. {result['file']}: {result['consistency_score']:.4f}")
    
    return {
        'avg_rand_score': avg_rand_score,
        'avg_ami_score': avg_ami_score,
        'avg_consistency_score': avg_consistency_score,
        'file_results': file_results
    }

def save_results_to_csv(results, output_path):
    """
    Save the directory analysis results to a CSV file
    """
    if results is None:
        print("No results to save.")
        return
    
    # Create a DataFrame for the file results
    df = pd.DataFrame(results['file_results'])
    
    # Add a row for the averages
    avg_row = pd.DataFrame([{
        'file': 'AVERAGE',
        'rand_score': results['avg_rand_score'],
        'ami_score': results['avg_ami_score'],
        'consistency_score': results['avg_consistency_score']
    }])
    
    df = pd.concat([df, avg_row], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def visualize_results(results):
    """
    Create visualizations of the directory analysis results
    """
    if results is None or not results['file_results']:
        print("No results to visualize.")
        return
    
    # Extract data
    files = [r['file'] for r in results['file_results']]
    consistency_scores = [r['consistency_score'] for r in results['file_results']]
    rand_scores = [r['rand_score'] for r in results['file_results']]
    ami_scores = [r['ami_score'] for r in results['file_results']]
    
    # Sort by consistency score
    sorted_indices = np.argsort(consistency_scores)[::-1]
    files = [files[i] for i in sorted_indices]
    consistency_scores = [consistency_scores[i] for i in sorted_indices]
    rand_scores = [rand_scores[i] for i in sorted_indices]
    ami_scores = [ami_scores[i] for i in sorted_indices]
    
    # Truncate filenames if they're too long
    files = [f[:25] + '...' if len(f) > 25 else f for f in files]
    
    # Plot consistency scores
    plt.figure(figsize=(12, 6))
    plt.bar(files, consistency_scores, color='royalblue')
    plt.axhline(y=results['avg_consistency_score'], color='r', linestyle='-', label=f"Avg: {results['avg_consistency_score']:.4f}")
    plt.title('Consistency Scores by File')
    plt.xlabel('File')
    plt.ylabel('Consistency Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Create a comparison bar chart of all metrics
    plt.figure(figsize=(12, 6))
    x = np.arange(len(files))
    width = 0.25
    
    plt.bar(x - width, consistency_scores, width, label='Consistency', color='royalblue')
    plt.bar(x, rand_scores, width, label='Adjusted Rand Index', color='forestgreen')
    plt.bar(x + width, ami_scores, width, label='Adjusted Mutual Info', color='darkorange')
    
    plt.title('Clustering Metrics by File')
    plt.xlabel('File')
    plt.ylabel('Score')
    plt.xticks(x, files, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python directory_analyzer.py <directory_path> [model_path] [file_pattern]")
        print("  directory_path: Path to directory containing MIDI pickle files")
        print("  model_path: Optional path to the model file (default: midi_transformer_consistency.pth)")
        print("  file_pattern: Optional file pattern to match (default: *.pkl)")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "midi_transformer_consistency-90.pth"
    file_pattern = sys.argv[3] if len(sys.argv) > 3 else "*.pkl"
    
    # Load the model
    model = load_model(model_path)
    
    # Process the directory
    results = process_directory(model, directory_path, file_pattern=file_pattern)
    
    if results:
        # Save results to CSV
        output_csv = os.path.join(directory_path, "consistency_results.csv")
        save_results_to_csv(results, output_csv)
        
        # Visualize the results
        visualize_results(results)
        
        print(f"\nOverall average consistency score: {results['avg_consistency_score']:.4f}")