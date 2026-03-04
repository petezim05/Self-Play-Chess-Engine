import chess
import numpy as np


PIECE_TO_CHANNEL = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


CHANNEL_TO_PIECE = {v: k for k, v in PIECE_TO_CHANNEL.items()}


def tensor_to_board(tensor: np.ndarray) -> chess.Board:
    """
    Reconstruct a chess.Board from an (18, 8, 8) tensor produced by board_to_tensor.

    Inverts the perspective flip applied when it is black's turn.
    """
    planes = tensor.copy()

    turn = chess.WHITE if planes[12, 0, 0] > 0.5 else chess.BLACK

    # Undo the perspective flip that board_to_tensor applied for black
    if turn == chess.BLACK:
        planes = np.flip(planes, axis=1).copy()

    board = chess.Board(fen=None)  # empty board, no pieces
    board.turn = turn

    # Place pieces
    for channel, (piece_type, color) in CHANNEL_TO_PIECE.items():
        for rank in range(8):
            for file in range(8):
                if planes[channel, rank, file] > 0.5:
                    square = chess.square(file, rank)
                    board.set_piece_at(square, chess.Piece(piece_type, color))

    # Castling rights
    castling = 0
    if planes[13, 0, 0] > 0.5:
        castling |= chess.BB_H1  # white kingside
    if planes[14, 0, 0] > 0.5:
        castling |= chess.BB_A1  # white queenside
    if planes[15, 0, 0] > 0.5:
        castling |= chess.BB_H8  # black kingside
    if planes[16, 0, 0] > 0.5:
        castling |= chess.BB_A8  # black queenside
    board.castling_rights = castling

    # En passant: file is marked in channel 17; rank is always 5 for white's turn,
    # 2 for black's turn (after undoing the perspective flip)
    ep_rank = 5 if turn == chess.WHITE else 2
    ep_cols = np.where(planes[17, ep_rank, :] > 0.5)[0]
    board.ep_square = chess.square(int(ep_cols[0]), ep_rank) if len(ep_cols) else None

    return board


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encode a chess board as an (18, 8, 8) float32 tensor suitable for CNN input.

    Channels:
        0-5:  White pieces  (P, N, B, R, Q, K)
        6-11: Black pieces  (P, N, B, R, Q, K)
        12:   Side to move  (all 1s = white, all 0s = black)
        13:   White kingside castling right
        14:   White queenside castling right
        15:   Black kingside castling right
        16:   Black queenside castling right
        17:   En passant file (1s along the target file, else 0)

    The board is always encoded from the current player's perspective
    (flipped vertically when it is black's turn).
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes
    for square, piece in board.piece_map().items():
        rank = square // 8
        file = square % 8
        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        planes[channel, rank, file] = 1.0

    # Side to move
    planes[12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights
    planes[13] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16] = float(board.has_queenside_castling_rights(chess.BLACK))

    # En passant
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        planes[17, :, ep_file] = 1.0

    # Normalise perspective: always encode from the current player's point of view
    if board.turn == chess.BLACK:
        planes = np.flip(planes, axis=1).copy()

    return planes


# --- Interactive game loop ---
"""
board = chess.Board()
print(board)

while True:
    tensor = board_to_tensor(board)
    print(f"\nBoard tensor shape: {tensor.shape}")

    try:
        move = input("\nEnter your move (SAN): ")
        board.push_san(move)
        print(board)
    except chess.IllegalMoveError:
        print("Illegal or incorrectly written move, try again.")
    except chess.InvalidMoveError:
        print("Invalid move format, try again.")
"""