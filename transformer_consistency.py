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

    pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    print(f"Found {len(pkl_files)} .pkl files.")

    for pkl_file in pkl_files:
        try:
            with open(os.path.join(folder_path, pkl_file), "rb") as f:
                df, channels = pickle.load(f)
                features = df[["Pitch", "Velocity", "StartTime", "Duration"]].values
                
                # Generate a unique UUID for this song and convert it to a string
                song_uuid = uuid.uuid4()
                song_id = str(song_uuid)

                all_features.extend(features)
                all_channels.extend(channels)
                all_song_ids.extend([song_id] * len(features))
        except Exception as e:
            print(f"Skipping {pkl_file} due to error: {e}")

    print(f"Total notes loaded: {len(all_features)}")

    # Shuffle
    songs = {}
    for i, song_id in enumerate(all_song_ids):
        if song_id not in songs:
            songs[song_id] = []
        songs[song_id].append((all_features[i], all_channels[i]))
    
    song_ids_list = list(songs.keys())
    random.shuffle(song_ids_list)
    
    all_features = []
    all_channels = []
    all_song_ids = []
    
    for song_id in song_ids_list:
        for feature, channel in songs[song_id]:
            all_features.append(feature)
            all_channels.append(channel)
            all_song_ids.append(song_id)
    
    features = torch.tensor(all_features, dtype=torch.float)
    features = normalize_features(features)  # Normalize input features

    unique_song_ids = list(set(all_song_ids))
    song_id_to_idx = {id: idx for idx, id in enumerate(unique_song_ids)}
    song_indices = [song_id_to_idx[id] for id in all_song_ids]
    
    if subset_size and subset_size < len(features):
        subset_indices = random.sample(range(len(features)), subset_size)
        features = features[subset_indices]
        channels = [all_channels[i] for i in subset_indices]
        song_indices = [song_indices[i] for i in subset_indices]
        return features, torch.tensor(channels, dtype=torch.long), torch.tensor(song_indices, dtype=torch.long)
    
    # Return features, channels, and integer indices for song IDs
    return features, torch.tensor(all_channels, dtype=torch.long), torch.tensor(song_indices, dtype=torch.long)

def train_model(folder_path, num_epochs=20, learning_rate=1e-5, batch_size=128, subset_size=None, consistency_weight=0.5):
    features, targets, song_indices = load_all_pkls_by_song(folder_path, subset_size=subset_size)
    
    # Create the model
    model = CrossAttentionTransformer(input_dim=4, model_dim=128, num_heads=16, num_layers=8, num_classes=16)
    
    # Move model to device (GPU or CPU)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(features)} notes...")
    print(f"Using consistency weight: {consistency_weight}")

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    unique_song_indices = torch.unique(song_indices)
    
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
            
            # Get all notes for this song
            song_mask = (song_indices == song_idx)
            song_features = features[song_mask]
            song_targets = targets[song_mask]
            
            # skip any bugged songs
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
                
                # Calculate classification loss
                classification_loss = criterion(logits, batch_targets)
                
                # Get predictions
                _, predicted = torch.max(logits, 1)
                
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
                # Lower consistency_weight means emphasize correct classifications more
                # Higher consistency_weight means emphasize clustering consistency more
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
        
        # Calculate average loss and metrics for the epoch
        avg_loss = total_loss / max(1, total_count)
        avg_consistency = total_consistency / max(1, total_count)
        avg_accuracy = total_accuracy / max(1, total_count)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Consistency: {avg_consistency:.4f} | Accuracy: {avg_accuracy:.4f}")

    print("Training finished.")
    return model

if __name__ == "__main__":
    pkl_folder = "output_small"
    # Set consistency_weight between 0 and 1:
    # 0 means only use classification loss
    # 1 means only use consistency reward
    # 0.5 is balanced between both
    trained_model = train_model(pkl_folder, num_epochs=20, subset_size=None, consistency_weight=0.9)
    torch.save(trained_model.state_dict(), "midi_transformer_consistency-90.pth")
    print("Model saved.")