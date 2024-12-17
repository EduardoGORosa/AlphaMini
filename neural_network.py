import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from utils import encode_state, load_move_mappings

class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels, num_moves):
        super(AlphaZeroNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_moves)

        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Policy head
        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

from utils import encode_pi

class ChessDataset(Dataset):
    def __init__(self, data_path, move_to_index):
        """
        Initializes the dataset by loading self-play data.

        Args:
            data_path (str): Path to the self-play data file.
            move_to_index (dict): Mapping from move to index.
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.move_to_index = move_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a data sample and encode the policy vector to a fixed size.

        Args:
            idx (int): Index of the data sample.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: state tensor, fixed-size pi tensor, outcome tensor
        """
        state, pi, outcome = self.data[idx]

        # Ensure the policy vector is fixed-size by encoding it
        pi_encoded = encode_pi([move for move in self.move_to_index], pi, self.move_to_index)

        state_tensor = encode_state(state)  # Shape: [13, 8, 8]
        pi_tensor = torch.tensor(pi_encoded, dtype=torch.float32)  # Fixed-size policy vector
        outcome_tensor = torch.tensor(outcome, dtype=torch.float32)  # Game result
        return state_tensor, pi_tensor, outcome_tensor

def load_model_for_inference(model_path, num_moves):
    model = AlphaZeroNet(input_channels=13, num_moves=num_moves)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def train_model(model, data_path, move_to_index, batch_size=64, epochs=10, lr=0.001):
    dataset = ChessDataset(data_path, move_to_index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for state, pi, outcome in dataloader:
            state, pi, outcome = state.to(device), pi.to(device), outcome.to(device)

            policy_logits, value = model(state)

            policy_loss = torch.nn.functional.cross_entropy(policy_logits, pi)
            value_loss = torch.nn.functional.mse_loss(value.squeeze(), outcome)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

    torch.save(model.state_dict(), "alpha_zero_model.pt")
    print("Model training complete and saved as alpha_zero_model.pt.")
