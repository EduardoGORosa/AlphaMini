# evaluate_ai.py

import chess
import chess.engine
import numpy as np
import trueskill
import os
from mcts import MCTS
import torch
import copy
from chess_env import ChessEnv

from neural_network import AlphaZeroNet, load_model_for_inference
from self_play import load_move_mappings

########################################
# 1) AI Implementation
########################################

def get_ai_move(board, mcts, num_simulations=800, batch_size=8):
    """
    Determines the AI's move using MCTS guided by the neural network.
    
    Parameters:
    - board: chess.Board object representing the current game state.
    - mcts: An instance of the MCTS class initialized with the neural network and move mappings.
    - num_simulations: Number of MCTS simulations to run.
    - batch_size: Number of simulations to process in a batch.
    
    Returns:
    - ai_move_uci: The selected move in UCI format.
    """
    
    # Run MCTS simulations
    mcts.run_simulations(num_simulations=num_simulations, batch_size=batch_size)
    
    # Retrieve visit counts from MCTS
    visit_counts = mcts.get_move_visit_counts()
    
    # Get all legal moves from the current board state
    legal_moves = list(board.legal_moves)
    
    # Extract visit counts for legal moves
    counts = np.array([visit_counts.get(move, 0) for move in legal_moves], dtype=np.float32)
    
    # Normalize the counts to get probabilities
    total_visits = counts.sum()
    if total_visits == 0:
        # Fallback: Assign equal probability if no visits recorded
        counts = np.ones_like(counts)
        total_visits = counts.sum()
    pi = counts / total_visits
    
    # Select the move with the highest visit count (greedy selection)
    best_idx = np.argmax(pi)
    ai_move = legal_moves[best_idx]
    
    # Return the move in UCI format
    return ai_move.uci()

########################################
# 2) Stockfish Opponent Setup
########################################

class StockfishOpponent:
    def __init__(self, stockfish_path, skill_level, name):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        # Lower skill level => weaker
        self.engine.configure({"Skill Level": skill_level})
        self.skill_level = skill_level
        self.name = name

    def get_move(self, board, time_limit=0.1):
        limit = chess.engine.Limit(time=time_limit)
        result = self.engine.play(board, limit)
        if result.move is None:
            return None
        return result.move.uci()

    def close(self):
        self.engine.quit()

########################################
# 3) TrueSkill-based Rating Approach
########################################

def update_rating(ai_rating, opp_rating, score, ts_env):
    """
    Score: 1.0 (AI win), 0.5 (draw), 0.0 (AI loss).
    Uses rate_1vs1(...) which does not accept 'ranks'.
    We'll handle the logic by controlling 'drawn'
    and who is first/second in the call.
    """
    if score == 1.0:
        # AI (rating1) wins, not a draw
        ai_new, opp_new = ts_env.rate_1vs1(ai_rating, opp_rating, drawn=False)
    elif score == 0.5:
        # It's a draw
        ai_new, opp_new = ts_env.rate_1vs1(ai_rating, opp_rating, drawn=True)
    else:  # score == 0.0 => AI lost
        # Opponent is winner
        opp_new, ai_new = ts_env.rate_1vs1(opp_rating, ai_rating, drawn=False)

    return ai_new, opp_new


def approximate_elo(trueskill_rating):
    """
    Convert a TrueSkill Rating to an approximate Elo for display.
    A common rough approach:
      Elo ~ 173.7178 * mu  ( if using default trueskill parameters ) 
    But you can tweak the factor or do a calibration match.
    """
    return int(173.7178 * trueskill_rating.mu)

########################################
# 4) Playing Multiple Games
########################################

def play_one_game(ai_as_white, stockfish_opponent, mcts, device='cpu', num_simulations=800, batch_size=8, time_limit=0.1):
    board = chess.Board()
    ai_color = chess.WHITE if ai_as_white else chess.BLACK

    while not board.is_game_over():
        if board.turn == ai_color:
            ai_move_uci = get_ai_move(board, mcts, num_simulations=num_simulations, batch_size=batch_size)
            if ai_move_uci is None:
                break
            move = chess.Move.from_uci(ai_move_uci)
            if move not in board.legal_moves:
                # Illegal move => immediate loss
                board.clear()
                break
            board.push(move)
        else:
            opp_move_uci = stockfish_opponent.get_move(board, time_limit)
            if opp_move_uci is None:
                break
            move = chess.Move.from_uci(opp_move_uci)
            if move not in board.legal_moves:
                board.clear()
                break
            board.push(move)
    return board.result()

########################################
# 5) Main Evaluation
########################################

def evaluate_ai(
    stockfish_path,
    opponents_info,
    games_per_opponent=10,
    time_limit=0.1,
    model_path="alpha_zero_model.pt",
    move_mapping_path="move_to_index.pkl",
    device='cpu',
    num_simulations=800,
    batch_size=8
):
    """
    opponents_info => list of (name, skill_level, approximate_elo).
    We'll create a TrueSkill environment and maintain a rating for AI + rating for each opponent.
    """
    # 1) Create TrueSkill environment
    # default mu=25.0, sigma=25/3, beta=25/6 => can be tuned
    ts_env = trueskill.TrueSkill(draw_probability=0.1)
    ai_rating = ts_env.create_rating()  # AI starts with mu=25, sigma=8.333

    # Load move mappings
    move_to_index, index_to_move = load_move_mappings(move_mapping_path)

    # Load the trained neural network model
    model = load_trained_model(model_path, device=device)
    env = ChessEnv()
    # Initialize MCTS
    mcts = MCTS(model=model, move_to_index=move_to_index, c_puct=1.4)
    mcts.initialize_root(env)
    # We'll store a rating for each opponent as well
    # We'll initialize each opponent with a 'fake' TrueSkill rating
    # that corresponds to their approximate Elo
    def from_elo_to(trueskill_env, elo):
        # Invert approximate_elo() => mu ~ elo / 173.7178
        mu = elo / 173.7178
        return trueskill_env.create_rating(mu=mu)

    # Build the opponents
    opponents = []
    for (opp_name, skill_level, opp_elo) in opponents_info:
        opp_ts = from_elo_to(ts_env, opp_elo)
        # store in a dict
        opponents.append({
            "name": opp_name,
            "skill_level": skill_level,
            "elo": opp_elo,
            "rating": opp_ts
        })

    # 2) For each opponent, play some games
    for opp in opponents:
        print(f"\n### Evaluating vs {opp['name']} (Skill={opp['skill_level']} ~ Elo={opp['elo']})")
        sfo = StockfishOpponent(stockfish_path, opp['skill_level'], opp['name'])

        for g in range(1, games_per_opponent+1):
            # We'll have AI always be White for simplicity
            result_str = play_one_game(
                ai_as_white=True,
                stockfish_opponent=sfo,
                mcts=mcts,
                device=device,
                num_simulations=num_simulations,
                batch_size=batch_size,
                time_limit=time_limit
            )
            # interpret result
            if result_str == "1-0":
                # AI (White) wins => score=1
                score = 1.0
            elif result_str == "0-1":
                score = 0.0
            else:
                # "1/2-1/2" => draw
                score = 0.5

            # update both ratings in TrueSkill
            ai_new, opp_new = update_rating(ai_rating, opp['rating'], score, ts_env)
            ai_rating, opp['rating'] = ai_new, opp_new

            # print partial results
            result_text = "win" if score==1.0 else "loss" if score==0.0 else "draw"
            ai_elo = approximate_elo(ai_rating)
            opp_elo_updated = approximate_elo(opp['rating'])
            print(f"Game {g}: AI {result_text}, new AI Elo ~ {ai_elo}, Opponent Elo ~ {opp_elo_updated}")

        sfo.close()

    # 3) Final AI rating
    final_elo = approximate_elo(ai_rating)
    return final_elo

########################################
# Helper Functions
########################################

def load_move_mappings(filepath="move_to_index.pkl"):
    """
    Loads the move_to_index mapping from a pickle file.
    
    Parameters:
    - filepath: Path to the move_to_index.pkl file.
    
    Returns:
    - move_to_index: Dictionary mapping move keys to indices.
    - index_to_move: Dictionary mapping indices to move keys.
    """
    import pickle
    with open(filepath, "rb") as f:
        move_to_index = pickle.load(f)
    # Create inverse mapping for easy lookup
    index_to_move = {v: k for k, v in move_to_index.items()}
    return move_to_index, index_to_move

def load_trained_model(model_path, device='cpu'):
    """
    Loads the trained neural network model for inference.
    
    Parameters:
    - model_path: Path to the saved model file.
    - device: Device to load the model on ('cpu' or 'cuda').
    
    Returns:
    - model: The loaded neural network model.
    """
    model = AlphaZeroNet(input_channels=13, num_res_blocks=20, dropout_rate=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

########################################
# Example Usage
########################################

def main():
    STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Update this path accordingly

    # Define opponents: (name, skill_level, approximate_elo)
    opponents_info = [
        ("SF_300Elo", 0, 300),
        ("SF_800Elo", 1, 800),
        ("SF_1200Elo", 5, 1200),
    ]

    # Evaluate
    final_elo = evaluate_ai(
        stockfish_path=STOCKFISH_PATH,
        opponents_info=opponents_info,
        games_per_opponent=10,
        time_limit=0.05,    # Extremely short time => degrade SF
        model_path="alpha_zero_model.pt",  # Path to your trained model
        move_mapping_path="move_to_index.pkl",  # Path to your move mappings
        device='cpu',       # Change to 'cuda' if using GPU
        num_simulations=800,  # Number of MCTS simulations per move
        batch_size=8        # Batch size for MCTS simulations
    )

    print(f"\n=== Final AI TrueSkill-based rating: ~{final_elo} Elo ===")

if __name__ == "__main__":
    main()
