import chess
import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.current_player = 1  # +1 for White, -1 for Black

    def reset(self):
        self.board.reset()
        self.current_player = 1
        return self.get_state_representation()

    def get_state_representation(self):
        """
        Converts the current board state into a multi-plane binary representation.
        Each piece type and color has its own plane.
        An additional plane indicates the current player's turn.
        """
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        planes = 13  # 12 for pieces + 1 for side to move
        state = np.zeros((planes, 8, 8), dtype=np.float32)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                plane_idx = piece_map[piece.symbol()]
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                state[plane_idx][row][col] = 1.0

        # Side-to-move plane
        # +1 if White, -1 if Black
        state[12] = self.current_player

        return state

    def step(self, move_uci):
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")

        self.board.push(move)
        self.current_player *= -1  # Switch side

        done = self.is_game_over()
        reward = self.get_result() if done else 0
        return self.get_state_representation(), done, reward

    def is_game_over(self):
        """
        We'll check standard 'is_game_over()' from python-chess,
        PLUS forcibly claim draw if 50-move or threefold repetition is possible.
        """
        # If python-chess says the game is over (checkmate, stalemate, etc.),
        # we're done:
        if self.board.is_game_over():
            return True

        # If 50 moves or threefold can be claimed, *force* a draw immediately:
        if self.board.can_claim_fifty_moves() or self.board.can_claim_threefold_repetition():
            return True
        
        return False
    
    def set_game_over(self):
        self._game_over = True

    def get_result(self):
        """
        Returns the game result from the perspective of the player who just moved:
         +1 for win, -1 for loss,  0 for draw.
        """
        # If it’s not truly over, no result:
        if not self.is_game_over():
            return 0

        # 1) If python-chess sees a standard checkmate/stalemate/insufficient:
        if self.board.is_game_over():
            res_str = self.board.result()  # e.g. '1-0', '0-1', or '1/2-1/2'
            if res_str == '1-0':
                # White won => from perspective of the last mover
                # If last mover = White => +1, else -1
                return 1 if self.current_player == -1 else -1
            elif res_str == '0-1':
                # Black won
                return 1 if self.current_player == 1 else -1
            else:
                # '1/2-1/2' => draw
                return 0

        # 2) If we can claim 50 moves or threefold => forced draw:
        if self.board.can_claim_fifty_moves() or self.board.can_claim_threefold_repetition():
            return 0  # draw

        # Should never get here if 'is_game_over' is consistent:
        return 0

    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]

    def render(self):
        print(self.board)

    def clone(self):
        cloned = ChessEnv()
        cloned.board = self.board.copy()
        cloned.current_player = self.current_player
        return cloned

    def get_last_move(self):
        if self.board.move_stack:
            return self.board.peek().uci()
        else:
            return None