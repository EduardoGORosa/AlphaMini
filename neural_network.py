##########################################
# neural_network.py
##########################################
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        bottleneck_channels = channels // 2
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=8, batch_first=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)
        
        # Apply Self-Attention
        batch, channels, height, width = out.size()
        out_reshaped = out.view(batch, channels, height * width).permute(0, 2, 1)  # [batch, seq_len, channels]
        attn_output, _ = self.attention(out_reshaped, out_reshaped, out_reshaped)
        attn_output = attn_output.permute(0, 2, 1).view(batch, channels, height, width)
        
        out = attn_output + residual
        out = self.relu(out)
        return out

########################################################
# Enhanced AlphaZeroNet with Increased Complexity
########################################################
class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels=13, num_res_blocks=20, dropout_rate=0.1):
        """
        Enhanced AlphaZero Network with increased depth, width, bottleneck residual blocks, dropout, and attention.
        """
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 512, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # Increased number of residual blocks and channels
        self.res_blocks = nn.ModuleList([
            ResidualBlock(512, dropout_rate) for _ in range(num_res_blocks)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(512, 4, kernel_size=1)  # Increased channels for policy
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_dropout = nn.Dropout(dropout_rate)
        self.policy_fc = nn.Linear(4 * 8 * 8, 20480)  # Adjusted for increased channels

        # Value Head
        self.value_conv = nn.Conv2d(512, 2, kernel_size=1)  # Increased channels for value
        self.value_bn = nn.BatchNorm2d(2)
        self.value_dropout = nn.Dropout(dropout_rate)
        self.value_fc1 = nn.Linear(2 * 8 * 8, 512)        # Increased size
        self.value_fc2 = nn.Linear(512, 256)              # Additional layer
        self.value_fc3 = nn.Linear(256, 1)                # Final output

        # Optional Global Attention Layer after Residual Blocks
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn(self.conv(x)))  # [batch, 512, 8, 8]

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)  # [batch, 512, 8, 8]

        # Optional Channel-wise Attention
        attn_weights = self.fc_attention(self.global_avg_pool(x).view(x.size(0), -1))
        attn_weights = attn_weights.view(x.size(0), 512, 1, 1)
        x = x * attn_weights

        # Policy Head
        p = self.relu(self.policy_bn(self.policy_conv(x)))    # [batch, 4, 8, 8]
        p = self.policy_dropout(p)
        p = p.reshape(p.size(0), -1)                          # [batch, 256]
        p = self.policy_fc(p)                                 # [batch, 40960]

        # Value Head
        v = self.relu(self.value_bn(self.value_conv(x)))      # [batch, 2, 8, 8]
        v = self.value_dropout(v)
        v = v.reshape(v.size(0), -1)                          # [batch, 128]
        v = F.relu(self.value_fc1(v))                        # [batch, 512]
        v = F.relu(self.value_fc2(v))                        # [batch, 256]
        v = torch.tanh(self.value_fc3(v))                    # [batch, 1]

        return p, v

########################################################
# Load/Save Model
########################################################
def initialize_and_save_model(model_path="alpha_zero_model.pt"):
    """
    Creates a brand-new net with policy=20480, saves it to 'model_path'.
    """
    model = AlphaZeroNet()
    torch.save(model.state_dict(), model_path)
    print(f"Initialized new model with policy=20480 => {model_path}")

def load_model_for_inference(model_path="alpha_zero_model.pt"):
    """
    Loads a net with policy=20480 from 'model_path'.
    If the checkpoint is from a net with dimension=2, it will mismatch.
    """
    model = AlphaZeroNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

########################################################
# ChessDataset + train_model
########################################################
class ChessDataset(Dataset):
    """
    Expects a .pkl with structure => 
      [ [ (state, pi, outcome), ...], [ (state, pi, outcome), ... ], ... ]
    We'll flatten into a single list of (state, pi, outcome).
    'pi' must have length 20480 for each position.
    """
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found.")

        with open(data_path, 'rb') as f:
            all_games = pickle.load(f)  # list-of-lists

        self.samples = []
        for game in all_games:
            self.samples.extend(game)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, pi, outcome = self.samples[idx]
        # state => [13,8,8]
        # pi => [20480]
        # outcome => +1/-1/0
        st_tensor = torch.tensor(state, dtype=torch.float32)
        pi_tensor = torch.tensor(pi, dtype=torch.float32)
        out_tensor = torch.tensor(outcome, dtype=torch.float32)
        return st_tensor, pi_tensor, out_tensor

def train_model(model, data_path, batch_size, epochs, lr):
    """
    Loads data from 'data_path', trains the net, saves as alpha_zero_model.pt
    """
    dataset = ChessDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for states, pis, outcomes in dataloader:
            states = states.to(device)
            pis = pis.to(device)
            outcomes = outcomes.to(device)

            optimizer.zero_grad()

            policy_logits, value = model(states)
            # [batch, 20480], [batch, 1]

            log_policy = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(pis * log_policy).sum(dim=1).mean()  # cross-entropy

            value = value.squeeze(dim=1)  # => [batch]
            value_loss = F.mse_loss(value, outcomes)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "alpha_zero_model.pt")
    print("Training complete => alpha_zero_model.pt saved.")
