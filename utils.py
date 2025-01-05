import numpy as np
import torch
import pickle
import os

def load_move_mappings(move_to_index_path="move_to_index.pkl"):
    """
    Read the dictionary move_uci->int from a .pkl
    => ensures we can embed the partial distribution into a 20480 vector
    """
    import pickle
    if not os.path.exists(move_to_index_path):
        raise FileNotFoundError(f"{move_to_index_path} not found")

    with open(move_to_index_path,"rb") as f:
        move_to_index = pickle.load(f)
    return move_to_index

def choose_move_from_pi(legal_moves, pi_distribution, temperature=1.0):
    """
    If temperature=0 => argmax, else sample from pi^1/t.
    """
    if temperature < 1e-6:
        return legal_moves[np.argmax(pi_distribution)]
    else:
        pi_temp = pi_distribution ** (1/temperature)
        pi_temp /= pi_temp.sum()
        return np.random.choice(legal_moves, p=pi_temp)

def encode_pi(legal_moves, pi, move_to_index):
    """
    Encodes the policy vector pi into a fixed-size vector based on move indices.
    """
    encoded_pi = np.zeros(len(move_to_index), dtype=np.float32)
    for move, prob in zip(legal_moves, pi):
        if move in move_to_index:
            encoded_pi[move_to_index[move]] = prob
    return encoded_pi

def decode_move(index, index_to_move):
    return index_to_move.get(index, None)

def encode_state(state_planes):
    # Convert the state representation into a PyTorch tensor.
    return torch.from_numpy(state_planes).float()

def encode_pi_tensor(encoded_pi):
    return torch.from_numpy(encoded_pi).float()
