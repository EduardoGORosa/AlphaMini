###############################################
# self_play.py
###############################################
import os
import pickle
import numpy as np
from multiprocessing import Pool
from chess_env import ChessEnv
from mcts import MCTS
from neural_network import load_model_for_inference
from utils import load_move_mappings, choose_move_from_pi


def play_single_game(args):
    """
    Worker function for multiprocessing.
    Returns [(state, pi, outcome), ...] for one completed self-play game.
    """
    (model_path, move_to_index_path, mcts_sims, temperature, game_id) = args

    # 1) Load the move dictionary => ~20480 entries
    move_to_index = load_move_mappings(move_to_index_path)
    num_moves = len(move_to_index)  # should be 20480

    # 2) Load the model => shape mismatch if old checkpoint had 2
    model = load_model_for_inference(model_path)

    # 3) Create a fresh environment
    env = ChessEnv()
    game_history = []
    move_count = 0
    mcts = MCTS(model, move_to_index)
    mcts.initialize_root(env)

    while not env.is_game_over():
        state = env.get_state_representation()  # [13,8,8]

        # MCTS => visit_counts
        mcts.run_simulations(num_simulations=mcts_sims, batch_size=8)
        visit_counts = visit_counts = mcts.get_move_visit_counts()

        legal_moves = env.get_legal_moves()
        counts = np.array([visit_counts.get(m, 0) for m in legal_moves], dtype=np.float32)
        if counts.sum() < 1e-6:
            counts = np.ones_like(counts)
        pi_distribution = counts / counts.sum()

        # embed partial distribution -> size=20480
        pi_encoded = np.zeros(num_moves, dtype=np.float32)
        for mv, prob in zip(legal_moves, pi_distribution):
            idx = move_to_index[mv]
            pi_encoded[idx] = prob

        game_history.append((state, pi_encoded, env.current_player))

        # temperature schedule
        temp = temperature if move_count < 10 else 0.0
        # pick move
        move_uci = choose_move_from_pi(legal_moves, pi_distribution, temp)
        env.step(move_uci)
        move_count += 1

        if move_count > 90:
            env.set_game_over()
            break

    # outcome
    final_result = env.get_result()  # +1, -1, or 0
    processed_game = []
    for (st, pi_enc, player) in game_history:
        # if player=1 => final_result; if -1 => -final_result
        outcome = final_result if player == 1 else -final_result
        processed_game.append((st, pi_enc, outcome))

    print(f"[Process] Game {game_id} ended. Moves={move_count}, Result={final_result}")
    return processed_game

def self_play_parallel(
    num_games=2,
    mcts_sims=50,
    model_path="alpha_zero_model.pt",
    move_to_index_path="move_to_index.pkl",
    temperature=1.0,
    output_data_path="self_play_data.pkl",
    num_processes=2
):
    """
    Runs 'num_games' self-play games in parallel. 
    Each game => list of (state, pi, outcome).
    We store a list-of-lists => [ game1, game2, ... ] in 'output_data_path'.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    if not os.path.exists(move_to_index_path):
        raise FileNotFoundError(f"move_to_index file '{move_to_index_path}' not found.")

    # Build argument list for each game
    args_list = []
    game_id = 1
    for _ in range(num_games):
        args_list.append((model_path, move_to_index_path, mcts_sims, temperature, game_id))
        game_id += 1

    results = []
    with Pool(processes=num_processes) as pool:
        for game_data in pool.imap_unordered(play_single_game, args_list):
            results.append(game_data)

    # save
    with open(output_data_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nParallel self-play done. {num_games} games => {output_data_path}")
