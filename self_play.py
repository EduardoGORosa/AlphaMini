import os
import pickle
import numpy as np
from chess_env import ChessEnv
from mcts import MCTS
from neural_network import load_model_for_inference, load_move_mappings
from utils import choose_move_from_pi
from multiprocessing import Pool, cpu_count

def play_single_game(args):
    """
    Simulates a single self-play game.

    Args:
        args: Tuple containing model, move_to_index, index_to_move, mcts_sims, temperature.

    Returns:
        List of game data [(state, pi, outcome), ...].
    """
    model, move_to_index, index_to_move, mcts_sims, temperature = args
    env = ChessEnv()
    mcts = MCTS(model, move_to_index, index_to_move)
    game_history = []  # List of (state, pi_distribution, player)

    while not env.is_game_over():
        state = env.get_state_representation()
        pi_distribution = mcts.search(env, simulations=mcts_sims)
        legal_moves = env.get_legal_moves()

        # Safely extract probabilities for legal moves
        pi = []
        for move in legal_moves:
            try:
                idx = move_to_index[move]  # Check if move exists in mapping
                pi.append(pi_distribution[idx])
            except (KeyError, IndexError):
                pi.append(1e-6)  # Assign small probability if move is invalid or missing

        pi = np.array(pi)
        pi = pi / pi.sum() if pi.sum() > 0 else np.ones_like(pi) / len(pi)

        # Choose a move
        move = choose_move_from_pi(legal_moves, pi, temperature)
        game_history.append((state, pi, env.current_player))
        env.step(move)

    result = env.get_result()
    # Assign final outcomes for the game
    processed_game = [
        (state, pi, result if player == 1 else -result)
        for (state, pi, player) in game_history
    ]
    return processed_game


def self_play_parallel(num_games, mcts_sims, model_path, output_data_path, temperature=1.0):
    """
    Conducts parallel self-play games to generate training data.

    Args:
        num_games (int): Number of games to simulate.
        mcts_sims (int): Number of MCTS simulations per move.
        model_path (str): Path to the neural network model.
        output_data_path (str): Path to save the generated data.
        temperature (float): Temperature parameter for move selection.
    """
    move_to_index, index_to_move = load_move_mappings()
    model = load_model_for_inference(model_path, len(move_to_index))
    model.share_memory()  # Allows the model to be shared between processes

    # Prepare arguments for parallel processing
    num_processes = cpu_count()
    games_per_process = num_games // num_processes
    extra_games = num_games % num_processes
    game_args = []

    for i in range(num_processes):
        num_games_for_this_process = games_per_process + (1 if i < extra_games else 0)
        for _ in range(num_games_for_this_process):
            game_args.append((model, move_to_index, index_to_move, mcts_sims, temperature))

    print(f"Starting {num_games} games with {num_processes} processes...")

    # Use multiprocessing Pool to parallelize self-play
    with Pool(processes=num_processes) as pool:
        results = pool.map(play_single_game, game_args)

    # Combine results and save to file
    data = [item for game_data in results for item in game_data]
    with open(output_data_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Self-play data saved to {output_data_path}")

if __name__ == "__main__":
    NUM_GAMES = 100        # Total number of games to simulate
    MCTS_SIMS = 100        # Number of MCTS simulations per move
    MODEL_PATH = "alpha_zero_model.pt"
    OUTPUT_DATA_PATH = "self_play_data.pkl"
    TEMPERATURE = 1.0      # Set temperature for move selection

    self_play_parallel(NUM_GAMES, MCTS_SIMS, MODEL_PATH, OUTPUT_DATA_PATH, TEMPERATURE)
