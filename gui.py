# gui.py
import pygame
import sys
import chess
from chess_env import ChessEnv
from mcts import MCTS
from neural_network import load_model_for_inference
from self_play import load_move_mappings
import numpy as np
# Constants
WIDTH, HEIGHT = 512, 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

def load_images():
    """
    Load piece images from 'images' folder, resizing them to SQ_SIZE x SQ_SIZE.
    E.g. images/ 'wP.png', 'wN.png', etc.
    """
    pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK',
              'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(
            pygame.image.load(f"images/{piece}.png"), (SQ_SIZE, SQ_SIZE)
        )

def draw_board(screen):
    """
    Draw an 8x8 board with alternating light/dark squares.
    """
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board):
    """
    Draw the current board's pieces using preloaded IMAGES.
    """
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_color = 'w' if piece.color == chess.WHITE else 'b'
            piece_type = piece.symbol().upper()  # 'P','N','B',...
            piece_name = piece_color + piece_type
            screen.blit(IMAGES[piece_name], pygame.Rect(
                chess.square_file(square) * SQ_SIZE,
                (7 - chess.square_rank(square)) * SQ_SIZE,
                SQ_SIZE, SQ_SIZE
            ))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    screen.fill(pygame.Color("white"))
    load_images()
    running = True
    
    # Create a fresh environment
    board_env = ChessEnv()  # Must be your implemented environment class
    
    # Load mappings + model
    move_to_index = load_move_mappings()  # from "move_to_index.pkl" etc.
    model = load_model_for_inference("alpha_zero_model.pt")
    # Create MCTS instance. We pass the same model each time.
    mcts = MCTS(model, move_to_index)
    mcts.initialize_root(board_env)
    selected_square = None
    valid_moves = []

    def square_from_mouse(pos):
        """
        Convert x,y pixel coordinates to a chess square index (0..63).
        """
        x, y = pos
        file = x // SQ_SIZE
        rank = 7 - (y // SQ_SIZE)
        return chess.square(file, rank)

    def promote_pawn():
        """
        If a human makes a pawn move to the 1st/8th rank, ask which piece to promote to.
        """
        print("Promotion! Choose piece: q (Queen), r (Rook), n (Knight), b (Bishop)")
        while True:
            choice = input("Promote to (q/r/n/b): ").strip().lower()
            if choice in ['q', 'r', 'n', 'b']:
                return choice
            print("Invalid choice. Please choose again.")

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                square = square_from_mouse(pos)

                # Let human control White
                if board_env.current_player == 1:
                    if selected_square is None:
                        # Select piece
                        piece = board_env.board.piece_at(square)
                        if piece and piece.color == chess.WHITE:
                            selected_square = square
                            # Filter valid moves from that piece
                            valid_moves = [
                                m for m in board_env.board.legal_moves
                                if m.from_square == square
                            ]
                    else:
                        # Attempt move
                        move = chess.Move(selected_square, square)
                        # Pawn promotion
                        origin_piece = board_env.board.piece_at(selected_square)
                        if origin_piece and origin_piece.piece_type == chess.PAWN:
                            if chess.square_rank(square) in [0, 7]:
                                promotion_choice = promote_pawn()
                                promo_map = {
                                    'q': chess.QUEEN,
                                    'r': chess.ROOK,
                                    'n': chess.KNIGHT,
                                    'b': chess.BISHOP
                                }
                                move.promotion = promo_map[promotion_choice]

                        if move in valid_moves:
                            board_env.step(move.uci())
                        selected_square = None
                        valid_moves = []

        # Draw board + pieces
        draw_board(screen)
        draw_pieces(screen, board_env.board)

        # Highlight selected piece
        if selected_square is not None:
            file = chess.square_file(selected_square)
            rank = chess.square_rank(selected_square)
            pygame.draw.rect(
                screen, (0, 255, 0, 50),
                pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE), 3
            )
            for mv in valid_moves:
                tf = chess.square_file(mv.to_square)
                tr = chess.square_rank(mv.to_square)
                # Draw a small circle to show possible moves
                pygame.draw.circle(
                    screen, (0, 255, 0),
                    (tf * SQ_SIZE + SQ_SIZE // 2, (7 - tr) * SQ_SIZE + SQ_SIZE // 2), 10
                )

        pygame.display.flip()
        clock.tick(MAX_FPS)

        # AI's turn => current_player == -1 (Black)
        if board_env.current_player == -1 and not board_env.is_game_over():
            print("AI is thinking...")
            mcts.run_simulations(num_simulations=800, batch_size=16)
            visit_counts = mcts.get_move_visit_counts()
            legal_moves = board_env.get_legal_moves()
            counts = np.array([visit_counts.get(m, 0) for m in legal_moves], dtype=np.float32)
            total_visits = sum(counts)
            if total_visits == 0:
                # fallback if MCTS found no expansions (very unlikely)
                counts = [1]*len(counts)
                total_visits = len(counts)
            pi = [c/total_visits for c in counts]
            # Argmax
            best_idx = max(range(len(pi)), key=lambda i: pi[i])
            ai_move = legal_moves[best_idx]
            print(f"AI played: {ai_move}")
            board_env.step(ai_move)

        # Check game over
        if board_env.is_game_over():
            result = board_env.get_result()
            if result == 1:
                print("White wins!")
            elif result == -1:
                print("Black wins!")
            else:
                print("It's a draw!")
            running = False

    # Optional: if you want to wait or do something at the end
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
