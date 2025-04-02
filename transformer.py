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

warnings.filterwarnings('ignore', category=DeprecationWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
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
        # x has shape [batch_size, 4] - split into individual features
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

        # Project to output classes
        logits = self.fc_out(final_hidden)

        return logits

class NoteTransformer(nn.Module):
    def __init__(self, input_dim=4, model_dim=64, num_heads=8, num_layers=4, num_classes=16, dropout=0.1):
        super(NoteTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, num_classes)

    def forward(self, src):
        """
        src: Tensor of shape [seq_len, batch_size, input_dim]
        Returns:
            logits: Tensor of shape [seq_len, batch_size, num_classes]
        """
        x = self.input_linear(src)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        logits = self.fc_out(x)
        return logits

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

def normalize_features(features):
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    return (features - mean) / (std + 1e-6)  # Add a small epsilon to avoid division by zero

def train_model(folder_path, num_epochs=20, learning_rate=1e-5, batch_size=128, subset_size=None):
    features, targets, song_indices = load_all_pkls_by_song(folder_path, subset_size=subset_size)

    # Create the model
    model = CrossAttentionTransformer(input_dim=4, model_dim=64, num_heads=8, num_layers=4, num_classes=16)

    # Move model to device (GPU or CPU)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(features)} notes...")

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
        correct = 0
        total = 0

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

                #
                # Some issues in training led me to add some NaN checks
                #

                # Check for NaNs in logits
                if torch.isnan(logits).any():
                    print(f"Warning: NaN detected in logits. Skipping batch.")
                    continue

                loss = criterion(logits, batch_targets)

                # Check for NaNs in loss
                if torch.isnan(loss).any():
                    print(f"Warning: NaN detected in loss. Skipping batch.")
                    continue

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_targets).sum().item()
                total += batch_targets.size(0)

                song_loss += loss.item()

            if num_batches > 0:
                avg_song_loss = song_loss / num_batches
                total_loss += avg_song_loss

        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(unique_song_indices)
        accuracy = 100 * correct / max(1, total)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    print("Training finished.")
    return model

if __name__ == "__main__":
   pkl_folder = "output_small"
   trained_model = train_model(pkl_folder, num_epochs=20, subset_size=None)
   torch.save(trained_model.state_dict(), "midi_transformer.pth")
   print("Model saved.")