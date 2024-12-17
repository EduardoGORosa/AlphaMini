import numpy as np
import torch
import pickle

def choose_move_from_pi(legal_moves, pi, temperature=1.0):
    """
    Selects a move based on the probability distribution pi.
    """
    if temperature == 0:
        move_index = np.argmax(pi)
    else:
        pi = pi ** (1 / temperature)
        pi = pi / np.sum(pi)
        move_index = np.random.choice(len(pi), p=pi)
    return legal_moves[move_index]

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
    """
    Decodes an index back to a move in UCI format.
    """
    return index_to_move.get(index, None)

def encode_state(state_planes):
    """
    Converts the state representation into a PyTorch tensor.
    """
    return torch.from_numpy(state_planes).float()

def encode_pi_tensor(encoded_pi):
    """
    Converts the policy vector into a PyTorch tensor.
    """
    return torch.from_numpy(encoded_pi).float()

def load_move_mappings():
    """
    Loads the move-to-index and index-to-move mappings from separate files.
    
    Returns:
        Tuple[dict, dict]: move_to_index and index_to_move dictionaries.
    """
    with open("move_to_index.pkl", "rb") as f:
        move_to_index = pickle.load(f)
    with open("index_to_move.pkl", "rb") as f:
        index_to_move = pickle.load(f)
    
    return move_to_index, index_to_move
