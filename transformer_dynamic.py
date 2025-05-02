import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import math
import os
import random
import warnings
import uuid
import numpy as np
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings('ignore', category=DeprecationWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PositionalEncoding(nn.Module):
    """
    Ensures that notes are treated as sequential data
    """
    def __init__(self, d_model, dropout=0.1, max_len=30000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        return self.dropout(x + self.pe[:x.size(0)])


class CrossAttentionLayer(nn.Module):
    """
    Ensure that the sequences of note data (so the pitch sequence, velocity sequence, etc) can interact with each other
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim)
        )
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.dropout(attn_output)
        query = self.layer_norm1(query + attn_output)

        # Feed-forward layer
        ffn_output = self.ffn(query)
        output = self.layer_norm2(query + ffn_output)
        return output


class CrossAttentionTransformer(nn.Module):
    def __init__(self, input_dim=4, model_dim=64, num_heads=8, num_layers=4, num_classes=16, dropout=0.1):
        super(CrossAttentionTransformer, self).__init__()
        # project each feature to the model dimension
        self.pitch_proj = nn.Linear(1, model_dim)
        self.velocity_proj = nn.Linear(1, model_dim)
        self.time_proj = nn.Linear(1, model_dim)
        self.duration_proj = nn.Linear(1, model_dim)
        
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        
        # cross-attention layers
        self.cross_attention_layers = nn.ModuleList([CrossAttentionLayer(model_dim, num_heads, dropout) for _ in range(num_layers)])
        
        # output projection
        self.fc_out = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # x has shape [batch_size, 4] (split into individual features)
        batch_size = x.size(0)
        
        pitch = x[:, 0:1]  
        velocity = x[:, 1:2]
        time = x[:, 2:3]
        duration = x[:, 3:4]
        
        pitch_emb = self.pitch_proj(pitch)
        velocity_emb = self.velocity_proj(velocity)
        time_emb = self.time_proj(time)
        duration_emb = self.duration_proj(duration)
        
        # Stack the features as a sequence: [seq_len=4, batch_size, model_dim]
        stacked_emb = torch.stack([pitch_emb, velocity_emb, time_emb, duration_emb], dim=0)
        
        stacked_emb = self.positional_encoding(stacked_emb)
        
        # Process through cross-attention layers
        hidden_states = stacked_emb
        for layer in self.cross_attention_layers:
            hidden_states = layer(hidden_states, hidden_states, hidden_states)
        
        final_hidden = hidden_states[-1]
        
        # Project to classes
        logits = self.fc_out(final_hidden)
        
        return logits


def normalize_features(features):
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    return (features - mean) / (std + 1e-6)


def calculate_consistency_score(targets, predictions):
    """
    Calculate clustering consistency score
    """
    targets_np = targets.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    # Initialize mapping dictionaries
    label_mapping = {}
    
    # Calculate consistency score
    consistency_count = 0
    total_count = 0
    
    for pred_label in np.unique(predictions_np):
        mask = (predictions_np == pred_label)
        true_labels_in_cluster = targets_np[mask]
        
        # Skip if no elements in this cluster
        if len(true_labels_in_cluster) == 0:
            continue
            
        # Find most common true label in this predicted cluster
        unique_true, counts = np.unique(true_labels_in_cluster, return_counts=True)
        most_common_true = unique_true[np.argmax(counts)]
        label_mapping[pred_label] = most_common_true
        
        # Count notes that match the most common label for this cluster
        consistency_count += np.sum(true_labels_in_cluster == most_common_true)
        total_count += len(true_labels_in_cluster)
    
    # Calculate consistency score
    consistency_score = consistency_count / total_count if total_count > 0 else 0
    
    return consistency_score


def load_all_pkls_by_song(folder_path, subset_size=None):
    all_features = []
    all_channels = []
    all_song_ids = []
    all_song_names = []  # Store original song names
    song_channel_maps = {}  # Maps song_id to its channel mapping
    song_channel_counts = {}  # Maps song_id to number of active channels
    max_channels_global = 0  # Track global maximum number of channels

    pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    print(f"Found {len(pkl_files)} .pkl files.")

    for pkl_file in pkl_files:
        try:
            song_name = pkl_file.replace('.pkl', '')
            with open(os.path.join(folder_path, pkl_file), "rb") as f:
                df, channels = pickle.load(f)
                features = df[["Pitch", "Velocity", "StartTime", "Duration"]].values
                
                # Generate a unique UUID for this song
                song_uuid = str(uuid.uuid4())
                
                # Find unique channels used in this song and create a mapping for this song
                unique_song_channels = sorted(set(channels))  # Sort to ensure consistent ordering
                channel_map = {ch: idx for idx, ch in enumerate(unique_song_channels)}
                num_channels = len(unique_song_channels)
                
                # Map actual channels to sequential indices for this song
                mapped_channels = [channel_map[ch] for ch in channels]
                
                # Update song-specific channel information
                song_channel_maps[song_uuid] = channel_map
                song_channel_counts[song_uuid] = num_channels
                max_channels_global = max(max_channels_global, num_channels)
                
                # Store the data
                all_features.extend(features)
                all_channels.extend(mapped_channels)  # Store mapped channels
                all_song_ids.extend([song_uuid] * len(features))
                all_song_names.extend([song_name] * len(features))
                
                print(f"Loaded '{song_name}' with {num_channels} active channels: {unique_song_channels}")
                
        except Exception as e:
            print(f"Skipping {pkl_file} due to error: {e}")

    print(f"Total notes loaded: {len(all_features)}")
    print(f"Maximum channels in any song: {max_channels_global}")

    # Group data by song
    songs = {}
    for i, song_id in enumerate(all_song_ids):
        if song_id not in songs:
            songs[song_id] = []
        songs[song_id].append((all_features[i], all_channels[i], all_song_names[i]))
    
    # Shuffle songs
    song_ids_list = list(songs.keys())
    random.shuffle(song_ids_list)
    
    # Reconstruct data in shuffled order
    all_features = []
    all_channels = []
    all_song_ids = []
    all_song_names = []
    
    for song_id in song_ids_list:
        for feature, channel, song_name in songs[song_id]:
            all_features.append(feature)
            all_channels.append(channel)
            all_song_ids.append(song_id)
            all_song_names.append(song_name)
    
    # Convert to tensors and normalize
    features = torch.tensor(all_features, dtype=torch.float)
    features = normalize_features(features)
    
    # Create song index mapping for lookup
    unique_song_ids = list(set(all_song_ids))
    song_id_to_idx = {id: idx for idx, id in enumerate(unique_song_ids)}
    song_indices = [song_id_to_idx[id] for id in all_song_ids]
    
    # Handle subsetting if requested
    if subset_size and subset_size < len(features):
        subset_indices = random.sample(range(len(features)), subset_size)
        features = features[subset_indices]
        channels = [all_channels[i] for i in subset_indices]
        song_indices = [song_indices[i] for i in subset_indices]
        song_names = [all_song_names[i] for i in subset_indices]
        return (features, torch.tensor(channels, dtype=torch.long), 
                torch.tensor(song_indices, dtype=torch.long), max_channels_global, 
                song_channel_counts, song_channel_maps, song_names)
    
    # Return all data
    return (features, torch.tensor(all_channels, dtype=torch.long), 
            torch.tensor(song_indices, dtype=torch.long), max_channels_global, 
            song_channel_counts, song_channel_maps, all_song_names)


def train_model(folder_path, num_epochs=20, learning_rate=1e-5, batch_size=128, subset_size=None, consistency_weight=0.5,  checkpoint_dir="checkpoints", resume_from=None):
    # Load data with song-specific channel information
    (features, targets, song_indices, max_channels, 
     song_channel_counts, song_channel_maps, song_names) = load_all_pkls_by_song(folder_path, subset_size=subset_size)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a mapping from song indices to song IDs for lookup during training
    unique_song_indices = torch.unique(song_indices).tolist()
    song_idx_to_id = {idx: song_id for idx, song_id in 
                      enumerate([list(song_channel_counts.keys())[i] for i in range(len(song_channel_counts))])}
    
    print(f"Training on {len(features)} notes across {len(unique_song_indices)} songs...")
    print(f"Maximum number of channels in any song: {max_channels}")
    print(f"Using consistency weight: {consistency_weight}")
    
    # Starting epoch (default to 0, will be updated if resuming)
    start_epoch = 0

    # Create model with the maximum number of channels
    model = CrossAttentionTransformer(
        input_dim=4, 
        model_dim=128, 
        num_heads=16, 
        num_layers=8, 
        num_classes=max_channels
    )
    
    # Move model to device (GPU or CPU)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize weights if not resuming
    if resume_from is None:
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.apply(init_weights)
    else:
        # Resume from checkpoint
        print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_consistency = 0
        total_accuracy = 0
        total_count = 0
        total_batches = 0
        
        # Shuffle songs for each epoch
        song_order = torch.randperm(len(unique_song_indices))
        
        for i in range(len(song_order)):
            song_idx = unique_song_indices[song_order[i]]
            song_id = song_idx_to_id.get(song_idx)
            
            if not song_id or song_id not in song_channel_counts:
                print(f"Warning: Song ID {song_id} not found in channel counts. Skipping.")
                continue
                
            # Get all notes for this song
            song_mask = (song_indices == song_idx)
            song_features = features[song_mask]
            song_targets = targets[song_mask]
            
            # Get channel count for this song
            num_channels_for_song = song_channel_counts[song_id]
            
            # Skip any bugged songs
            if len(song_features) < 10:  # Minimum batch size threshold
                continue
                
            # Process in batches
            num_batches = (len(song_features) + batch_size - 1) // batch_size
            song_loss = 0
            song_consistency = 0
            song_accuracy = 0
            song_batch_count = 0
            
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx = min((j + 1) * batch_size, len(song_features))
                
                if start_idx >= end_idx:
                    continue
                    
                batch_features = song_features[start_idx:end_idx].to(device)
                batch_targets = song_targets[start_idx:end_idx].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(batch_features)
                
                # Check for NaNs in logits
                if torch.isnan(logits).any():
                    print(f"Warning: NaN detected in logits. Skipping batch.")
                    continue
                
                # Mask logits to only consider valid channels for this song
                # This is key: we limit predictions to only the channels this song actually uses
                if num_channels_for_song < max_channels:
                    mask = torch.zeros_like(logits)
                    mask[:, :num_channels_for_song] = 1.0
                    masked_logits = logits * mask + (1 - mask) * -1e9  # Set unused channels to large negative value
                else:
                    masked_logits = logits
                
                # Calculate classification loss on masked logits
                classification_loss = criterion(masked_logits, batch_targets)
                
                # Get predictions
                _, predicted = torch.max(masked_logits, 1)
                
                # Calculate traditional accuracy
                accuracy = (predicted == batch_targets).float().mean().item()
                
                # Calculate clustering consistency
                if len(torch.unique(predicted)) > 1 and len(torch.unique(batch_targets)) > 1:
                    consistency = calculate_consistency_score(batch_targets, predicted)
                else:
                    # If only one class is predicted or present in targets, set consistency
                    # based on whether predictions match or not
                    consistency = 1.0 if torch.all(predicted == batch_targets) else 0.0
                
                # Combined loss: regular loss - consistency reward
                combined_loss = (1 - consistency_weight) * classification_loss - consistency_weight * torch.tensor(consistency).to(device)
                
                # Check for NaNs in loss
                if torch.isnan(combined_loss).any():
                    print(f"Warning: NaN detected in combined loss. Skipping batch.")
                    continue
                
                # Backward pass and optimization
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
                song_loss += combined_loss.item()
                song_consistency += consistency
                song_accuracy += accuracy
                song_batch_count += 1
            
            if song_batch_count > 0:
                avg_song_loss = song_loss / song_batch_count
                avg_song_consistency = song_consistency / song_batch_count
                avg_song_accuracy = song_accuracy / song_batch_count
                
                total_loss += avg_song_loss
                total_consistency += avg_song_consistency
                total_accuracy += avg_song_accuracy
                total_count += 1
                total_batches += song_batch_count
                
                # Print song-specific metrics occasionally
                if i % 10 == 0:
                    print(f"Song {i+1}/{len(song_order)} | Channels: {num_channels_for_song} | "
                          f"Acc: {avg_song_accuracy:.4f} | Cons: {avg_song_consistency:.4f}")
        
        # Calculate average loss and metrics for the epoch
        avg_loss = total_loss / max(1, total_count)
        avg_consistency = total_consistency / max(1, total_count)
        avg_accuracy = total_accuracy / max(1, total_count)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"Consistency: {avg_consistency:.4f} | Accuracy: {avg_accuracy:.4f}")
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'consistency': avg_consistency,
            'accuracy': avg_accuracy,
            'max_channels': max_channels,
            'song_channel_counts': song_channel_counts,
            'song_channel_maps': song_channel_maps
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


    print("Training finished.")
    
    # Save channel mapping information along with the model
    model_info = {
        'max_channels': max_channels,
        'song_channel_counts': song_channel_counts,
        'song_channel_maps': song_channel_maps
    }
    
    return model, model_info


def evaluate_model(model, model_info, folder_path, batch_size=128):
    """
    Evaluate the model on data
    """
    # Load evaluation data
    (features, targets, song_indices, _, 
     song_channel_counts, song_channel_maps, song_names) = load_all_pkls_by_song(folder_path)
    
    # Create mapping for song indices
    unique_song_indices = torch.unique(song_indices).tolist()
    song_idx_to_id = {idx: song_id for idx, song_id in 
                      enumerate([list(song_channel_counts.keys())[i] for i in range(len(song_channel_counts))])}
    
    model.eval()
    total_accuracy = 0
    total_consistency = 0
    total_songs = 0
    
    # Group results by song
    song_metrics = {}
    
    with torch.no_grad():
        for i, song_idx in enumerate(unique_song_indices):
            song_id = song_idx_to_id.get(song_idx)
            
            if not song_id or song_id not in song_channel_counts:
                continue
                
            # Get all notes for this song
            song_mask = (song_indices == song_idx)
            song_features = features[song_mask]
            song_targets = targets[song_mask]
            song_name = song_names[song_mask.nonzero()[0].item()]
            
            # Get channel count for this song
            num_channels_for_song = song_channel_counts[song_id]
            
            all_predictions = []
            all_targets = []
            
            # Process in batches
            for j in range(0, len(song_features), batch_size):
                batch_features = song_features[j:j+batch_size].to(device)
                batch_targets = song_targets[j:j+batch_size].to(device)
                
                # Forward pass
                logits = model(batch_features)
                
                # Mask logits to consider only valid channels for this song
                if num_channels_for_song < model_info['max_channels']:
                    mask = torch.zeros_like(logits)
                    mask[:, :num_channels_for_song] = 1.0
                    masked_logits = logits * mask + (1 - mask) * -1e9
                else:
                    masked_logits = logits
                
                # Get predictions
                _, predicted = torch.max(masked_logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
            
            # Calculate metrics
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            accuracy = np.mean(all_predictions == all_targets)
            consistency = calculate_consistency_score(torch.tensor(all_targets), torch.tensor(all_predictions))
            rand_score = adjusted_rand_score(all_targets, all_predictions)
            
            # Store metrics for this song
            song_metrics[song_name] = {
                'accuracy': accuracy,
                'consistency': consistency,
                'rand_score': rand_score,
                'num_channels': num_channels_for_song,
                'num_notes': len(all_targets)
            }
            
            total_accuracy += accuracy
            total_consistency += consistency
            total_songs += 1
            
            if i % 10 == 0:
                print(f"Evaluated song {i+1}/{len(unique_song_indices)}: {song_name} | "
                      f"Channels: {num_channels_for_song} | Acc: {accuracy:.4f} | Cons: {consistency:.4f}")
    
    # Calculate overall metrics
    avg_accuracy = total_accuracy / total_songs if total_songs > 0 else 0
    avg_consistency = total_consistency / total_songs if total_songs > 0 else 0
    
    print(f"Overall Evaluation Results:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Consistency: {avg_consistency:.4f}")
    
    return song_metrics


def save_model_with_info(model, model_info, filename):
    """
    Save model along with channel mapping information
    """
    model_state = {
        'state_dict': model.state_dict(),
        'model_info': model_info
    }
    torch.save(model_state, filename)
    print(f"Model and metadata saved to {filename}")


def load_model_with_info(filename, model_class=CrossAttentionTransformer):
    """
    Load model along with its metadata
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")
        
    model_state = torch.load(filename, map_location=device)
    
    # Extract model info
    model_info = model_state.get('model_info', {})
    max_channels = model_info.get('max_channels', 16)  # Default to 16 if not found
    
    # Create model with the correct number of output classes
    model = model_class(
        input_dim=4, 
        model_dim=128, 
        num_heads=16, 
        num_layers=8, 
        num_classes=max_channels
    )
    
    # Load state dict
    model.load_state_dict(model_state['state_dict'])
    model.to(device)
    
    return model, model_info


if __name__ == "__main__":
    pkl_folder = "training_set_tiny"
    
    # Train model with dynamic channel handling
    trained_model, model_info = train_model(
        pkl_folder, 
        num_epochs=20, 
        subset_size=None, 
        consistency_weight=0.95
    )
    
    # Save model with metadata
    save_model_with_info(trained_model, model_info, "midi_transformer_dynamic_channels.pth")
    
    # Optional: Evaluate the model
    song_metrics = evaluate_model(trained_model, model_info, pkl_folder)
    
    # Save evaluation results
    with open("evaluation_results.pkl", "wb") as f:
        pickle.dump(song_metrics, f)
    
    print("Model saved with dynamic channel information.")