import torch
import torch.nn as nn
import pickle
import pandas as pd
import os
import random
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from collections import defaultdict

from transformer import CrossAttentionTransformer, normalize_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path, input_dim=4, model_dim=64, num_heads=8, num_layers=4, num_classes=16):
    """
    Load the trained CrossAttentionTransformer
    """
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

    label_mapping = {}
    consistency_count = 0
    total_count = 0

    for pred_label in np.unique(pred_labels):
        mask = (pred_labels == pred_label)
        true_labels_in_cluster = true_labels[mask]
        unique_true, counts = np.unique(true_labels_in_cluster, return_counts=True)
        most_common_true = unique_true[np.argmax(counts)]
        label_mapping[pred_label] = most_common_true
        
        consistency_count += np.sum(true_labels_in_cluster == most_common_true)
        total_count += len(true_labels_in_cluster)
    
    consistency_score = consistency_count / total_count if total_count > 0 else 0
    
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
    
    return rand_score, ami_score, consistency_score, predictions

def analyze_feature_impact_on_consistency(model, pkl_path, num_samples=100):
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
    
    # Idk wtf this does
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

if __name__ == "__main__":
    model_path = "midi_transformer_tiny.pth"
    test_file = "output_midi/Margaritaville.pkl"
    
    model = load_model(model_path)
    
    rand_score, ami_score, consistency_score, predictions = test_model_consistency(model, test_file)
    
    feature_impact = analyze_feature_impact_on_consistency(model, test_file, 2000)