# gui.py
import pygame
import sys
import chess
from chess_env import ChessEnv
from mcts import MCTS
from neural_network import load_model_for_inference, load_move_mappings
from utils import decode_move

# Constants
WIDTH, HEIGHT = 512, 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

def load_images():
    """
    Loads images for all chess pieces.
    Ensure that you have images named like 'wp.png', 'wN.png', etc.
    Place them in an 'images' directory.
    """
    pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK',
              'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(pygame.image.load(f"images/{piece}.png"), (SQ_SIZE, SQ_SIZE))

def draw_board(screen):
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_color = 'w' if piece.color == chess.WHITE else 'b'
            piece_type = piece.symbol().upper()  # Ensure lowercase for consistency
            piece_name = piece_color + piece_type  # Construct key like 'wP' or 'bq'
            screen.blit(IMAGES[piece_name], pygame.Rect(
                chess.square_file(square) * SQ_SIZE,
                (7 - chess.square_rank(square)) * SQ_SIZE,
                SQ_SIZE,
                SQ_SIZE
            ))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    screen.fill(pygame.Color("white"))
    load_images()
    running = True
    board_env = ChessEnv()
    move_to_index, index_to_move = load_move_mappings()
    model = load_model_for_inference("alpha_zero_model.pt", len(move_to_index))
    mcts = MCTS(model, move_to_index, index_to_move)

    selected_square = None  # Tracks the currently selected square
    valid_moves = []        # Stores valid moves for the selected piece

    def square_from_mouse(pos):
        """Convert mouse position to board square."""
        x, y = pos
        file = x // SQ_SIZE
        rank = 7 - (y // SQ_SIZE)
        return chess.square(file, rank)

    def promote_pawn():
        """Prompt for pawn promotion choice."""
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

                # Handle player input
                if board_env.current_player == 1:
                    if selected_square is None:
                        # Select piece
                        piece = board_env.board.piece_at(square)
                        if piece and piece.color == chess.WHITE:
                            selected_square = square
                            valid_moves = [
                                move for move in board_env.board.legal_moves
                                if move.from_square == square
                            ]
                    else:
                        # Attempt move
                        move = chess.Move(selected_square, square)
                        
                        # Handle pawn promotion
                        if board_env.board.piece_at(selected_square).piece_type == chess.PAWN:
                            if chess.square_rank(square) in [0, 7]:  # Promotion rank
                                promotion_choice = promote_pawn()
                                move.promotion = {'q': chess.QUEEN, 'r': chess.ROOK,
                                                  'n': chess.KNIGHT, 'b': chess.BISHOP}[promotion_choice]

                        if move in valid_moves:
                            board_env.step(move.uci())
                            selected_square = None
                            valid_moves = []
                        else:
                            selected_square = None  # Deselect on invalid move

        # Draw board and pieces
        draw_board(screen)
        draw_pieces(screen, board_env.board)

        # Highlight selected piece and valid moves
        if selected_square is not None:
            file = chess.square_file(selected_square)
            rank = chess.square_rank(selected_square)
            pygame.draw.rect(
                screen, (0, 255, 0, 50),
                pygame.Rect(file * SQ_SIZE, (7 - rank) * SQ_SIZE, SQ_SIZE, SQ_SIZE), 3
            )
            for move in valid_moves:
                target_file = chess.square_file(move.to_square)
                target_rank = chess.square_rank(move.to_square)
                pygame.draw.circle(
                    screen, (0, 255, 0),
                    (target_file * SQ_SIZE + SQ_SIZE // 2, (7 - target_rank) * SQ_SIZE + SQ_SIZE // 2), 10
                )

        pygame.display.flip()
        clock.tick(MAX_FPS)

        # AI's turn
        if board_env.current_player == -1 and not board_env.is_game_over():
            print("AI is thinking...")
            ai_move = mcts.search(board_env, simulations=100)
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


if __name__ == "__main__":
    main()
