import random
import joblib
import chess
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDRegressor

MODEL_PATH = Path("chess_model.joblib")
DATA_PATH = Path("training_data.csv")

PIECES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING
]

def encode_board(board):
    features = []

    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in PIECES:
            bitboard = board.pieces(piece_type, color)
            for square in chess.SQUARES:
                features.append(1 if square in bitboard else 0)

    features.append(1 if board.turn == chess.WHITE else -1)
    return np.array(features, dtype=np.float32)

def get_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    model = SGDRegressor(max_iter=1000, learning_rate="adaptive", eta0=0.01)
    dummy_x = np.zeros((1, 769), dtype=np.float32)
    dummy_y = np.array([0.0])
    model.partial_fit(dummy_x, dummy_y)
    return model

def material_score(board):
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    score = 0
    for piece_type, value in values.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    return score / 39

def evaluate(board, model):
    if board.is_checkmate():
        return -999 if board.turn == chess.WHITE else 999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    nn_score = model.predict([encode_board(board)])[0]
    return 0.7 * nn_score + 0.3 * material_score(board)

def choose_bot_move(board, model):
    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)

    best_move = None
    best_score = -999999 if board.turn == chess.WHITE else 999999

    for move in legal_moves:
        board.push(move)
        score = evaluate(board, model)
        board.pop()

        if board.turn == chess.WHITE and score > best_score:
            best_score = score
            best_move = move

        if board.turn == chess.BLACK and score < best_score:
            best_score = score
            best_move = move

    return best_move

def save_training_positions(positions, result):
    if result == "1-0":
        target = 1
    elif result == "0-1":
        target = -1
    else:
        target = 0

    with DATA_PATH.open("a") as f:
        for fen in positions:
            f.write(f"{fen}|{target}\n")

def train_model():
    if not DATA_PATH.exists():
        return

    xs = []
    ys = []

    with DATA_PATH.open() as f:
        for line in f:
            fen, target = line.strip().split("|")
            board = chess.Board(fen)
            xs.append(encode_board(board))
            ys.append(float(target))

    if not xs:
        return

    model = get_model()
    model.partial_fit(np.array(xs), np.array(ys))
    joblib.dump(model, MODEL_PATH)

def play_game():
    model = get_model()
    board = chess.Board()
    positions = []

    human_color = input("Play as white or black? ").strip().lower()
    human_is_white = human_color.startswith("w")

    while not board.is_game_over():
        print("\\n" + str(board))
        print("FEN:", board.fen())

        positions.append(board.fen())

        human_turn = board.turn == chess.WHITE if human_is_white else board.turn == chess.BLACK

        if human_turn:
            move_text = input("Your move: ").strip()

            try:
                move = board.parse_san(move_text)
            except ValueError:
                try:
                    move = chess.Move.from_uci(move_text)
                    if move not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("Invalid move. Try SAN like Nf3 or UCI like g1f3.")
                    continue

            board.push(move)
        else:
            move = choose_bot_move(board, model)
            print("Bot plays:", board.san(move))
            board.push(move)

    print("\\nFinal board:")
    print(board)
    print("Result:", board.result())

    save_training_positions(positions, board.result())
    train_model()
    print("Saved game and updated model.")

if __name__ == "__main__":
    play_game()
