# move_encoding.py
import chess
import pickle

def generate_move_to_index():
    """
    Generates a mapping from all possible UCI moves to unique indices.
    """
    move_to_index = {}
    index = 0
    # Iterate over all from and to squares
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            # Normal moves
            move = chess.Move(from_square, to_square)
            move_to_index[move.uci()] = index
            index += 1

            # Promotions
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move_promo = chess.Move(from_square, to_square, promotion=promotion)
                move_to_index[move_promo.uci()] = index
                index += 1

    return move_to_index

def generate_index_to_move(move_to_index):
    """
    Generates the inverse mapping from indices to UCI moves.
    """
    return {v: k for k, v in move_to_index.items()}

if __name__ == "__main__":
    move_to_index = generate_move_to_index()
    index_to_move = generate_index_to_move(move_to_index)

    # Save mappings for later use
    with open("move_to_index.pkl", "wb") as f:
        pickle.dump(move_to_index, f)

    with open("index_to_move.pkl", "wb") as f:
        pickle.dump(index_to_move, f)

    print(f"Total moves encoded: {len(move_to_index)}")
