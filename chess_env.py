# chess_env.py
import chess
import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.current_player = 1  # 1 for White, -1 for Black

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
        planes = 13  # 12 for pieces + 1 for turn
        state = np.zeros((planes, 8, 8), dtype=np.float32)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                plane = piece_map[piece.symbol()]
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                state[plane][row][col] = 1.0

        # Turn plane
        state[12] = self.current_player  # 1 for White, -1 for Black

        return state

    def step(self, move_uci):
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")

        self.board.push(move)
        self.current_player *= -1  # Switch player

        done = self.is_game_over()
        reward = self.get_result() if done else 0

        return self.get_state_representation(), done, reward

    def is_game_over(self):
        return self.board.is_game_over()

    def get_result(self):
        """
        Returns the game result from the perspective of the player who just moved.
        +1 for win, -1 for loss, 0 for draw.
        """
        if not self.board.is_game_over():
            return 0

        result = self.board.result()
        if result == '1-0':
            return 1 if self.current_player == -1 else -1
        elif result == '0-1':
            return 1 if self.current_player == 1 else -1
        else:
            return 0  # Draw

    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]

    def render(self):
        print(self.board)

    def clone(self):
        cloned = ChessEnv()
        cloned.board = self.board.copy()
        cloned.current_player = self.current_player
        return cloned
