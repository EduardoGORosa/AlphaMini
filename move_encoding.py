import chess
import pickle

def generate_move_to_index():
    """
    Enumerates all possible from-square -> to-square combos (64 x 64 = 4096)
    PLUS promotions (4 possible promotion pieces).
    => 64 x 64 x 5 = 20,480 total move strings in 'move_to_index'.
    """
    move_to_index = {}
    idx = 0

    for from_sq in chess.SQUARES:  # 0..63
        for to_sq in chess.SQUARES:  # 0..63
            # 1) Normal move (no promotion)
            normal_move = chess.Move(from_sq, to_sq)
            move_uci = normal_move.uci()  # e.g. "e2e4"
            move_to_index[move_uci] = idx
            idx += 1

            # 2) Promotion moves (Q,R,B,N)
            for promo_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_sq, to_sq, promotion=promo_piece)
                promo_uci = promo_move.uci()  # e.g. "e7e8q"
                move_to_index[promo_uci] = idx
                idx += 1

    return move_to_index

if __name__ == "__main__":
    move_to_index = generate_move_to_index()
    print("Number of moves enumerated:", len(move_to_index))  # Should print 20480
    with open("move_to_index.pkl", "wb") as f:
        pickle.dump(move_to_index, f)

    print(f"move_to_index.pkl saved with dimension: {len(move_to_index)}")
