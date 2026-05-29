import random
from pathlib import Path

import chess
import joblib
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.linear_model import SGDRegressor


MODEL_PATH = Path("chess_model.joblib")
DATA_PATH = Path("training_data.csv")

PIECES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING
]

PIECE_SYMBOLS = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


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
    model.partial_fit(np.zeros((1, 769), dtype=np.float32), np.array([0.0]))
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
        elif board.turn == chess.BLACK and score < best_score:
            best_score = score
            best_move = move

    return best_move or random.choice(legal_moves)


def save_training_positions(positions, result):
    if result == "1-0":
        target = 1
    elif result == "0-1":
        target = -1
    else:
        target = 0

    with DATA_PATH.open("a", encoding="utf-8") as f:
        for fen in positions:
            f.write(f"{fen}|{target}\n")


def train_model():
    if not DATA_PATH.exists():
        return

    xs = []
    ys = []

    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            if "|" not in line:
                continue
            fen, target = line.strip().split("|")
            board = chess.Board(fen)
            xs.append(encode_board(board))
            ys.append(float(target))

    if not xs:
        return

    model = get_model()
    model.partial_fit(np.array(xs), np.array(ys))
    joblib.dump(model, MODEL_PATH)


class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Learning Chess Bot - Resizable")
        self.root.geometry("720x780")
        self.root.minsize(360, 420)

        self.board = chess.Board()
        self.model = get_model()
        self.positions = []
        self.selected_square = None
        self.human_is_white = True

        self.status = tk.Label(root, text="Choose your side", font=("Arial", 14))
        self.status.pack(pady=8)

        side_frame = tk.Frame(root)
        side_frame.pack()

        tk.Button(side_frame, text="Play as White", command=lambda: self.start_game(True)).pack(side=tk.LEFT, padx=5)
        tk.Button(side_frame, text="Play as Black", command=lambda: self.start_game(False)).pack(side=tk.LEFT, padx=5)
        tk.Button(side_frame, text="New Game", command=self.new_game).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(root, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas.bind("<Configure>", lambda event: self.draw_board())
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.draw_board()

    def start_game(self, human_is_white):
        self.human_is_white = human_is_white
        self.board = chess.Board()
        self.positions = []
        self.selected_square = None

        if self.human_is_white:
            self.status.config(text="Your turn. Click a piece, then click where to move.")
        else:
            self.status.config(text="Bot thinking...")
            self.root.after(300, self.bot_move)

        self.draw_board()

    def new_game(self):
        self.start_game(self.human_is_white)

    def board_geometry(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        size = max(1, min(width, height))
        x0 = (width - size) // 2
        y0 = (height - size) // 2
        square_size = size / 8
        return x0, y0, size, square_size

    def square_from_row_col(self, row, col):
        if self.human_is_white:
            rank = 7 - row
            file = col
        else:
            rank = row
            file = 7 - col
        return chess.square(file, rank)

    def row_col_from_square(self, square):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if self.human_is_white:
            row = 7 - rank
            col = file
        else:
            row = rank
            col = 7 - file

        return row, col

    def draw_board(self):
        self.canvas.delete("all")
        x0, y0, size, square_size = self.board_geometry()

        light = "#f0d9b5"
        dark = "#b58863"
        selected = "#f6f669"

        for row in range(8):
            for col in range(8):
                square = self.square_from_row_col(row, col)
                color = light if (row + col) % 2 == 0 else dark
                if self.selected_square == square:
                    color = selected

                x1 = x0 + col * square_size
                y1 = y0 + row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

                piece = self.board.piece_at(square)
                if piece:
                    symbol = PIECE_SYMBOLS[piece.symbol()]
                    font_size = max(14, int(square_size * 0.62))
                    self.canvas.create_text(
                        x1 + square_size / 2,
                        y1 + square_size / 2,
                        text=symbol,
                        font=("Arial", font_size),
                    )

    def is_human_turn(self):
        return self.board.turn == chess.WHITE if self.human_is_white else self.board.turn == chess.BLACK

    def on_canvas_click(self, event):
        if self.board.is_game_over() or not self.is_human_turn():
            return

        x0, y0, size, square_size = self.board_geometry()

        if not (x0 <= event.x <= x0 + size and y0 <= event.y <= y0 + size):
            return

        col = int((event.x - x0) // square_size)
        row = int((event.y - y0) // square_size)

        if row < 0 or row > 7 or col < 0 or col > 7:
            return

        square = self.square_from_row_col(row, col)
        piece = self.board.piece_at(square)

        if self.selected_square is None:
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.draw_board()
            return

        move = chess.Move(self.selected_square, square)

        moving_piece = self.board.piece_at(self.selected_square)
        if moving_piece and moving_piece.piece_type == chess.PAWN:
            target_rank = chess.square_rank(square)
            if target_rank == 0 or target_rank == 7:
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)

        if move in self.board.legal_moves:
            self.positions.append(self.board.fen())
            self.board.push(move)
            self.selected_square = None
            self.draw_board()
            self.check_game_over()

            if not self.board.is_game_over():
                self.status.config(text="Bot thinking...")
                self.root.after(300, self.bot_move)
        else:
            self.selected_square = None
            self.draw_board()

    def bot_move(self):
        if self.board.is_game_over():
            return

        self.positions.append(self.board.fen())
        move = choose_bot_move(self.board, self.model)
        san = self.board.san(move)
        self.board.push(move)

        self.status.config(text=f"Bot played {san}. Your turn.")
        self.draw_board()
        self.check_game_over()

    def check_game_over(self):
        if not self.board.is_game_over():
            return

        result = self.board.result()
        save_training_positions(self.positions, result)
        train_model()

        if self.board.is_checkmate():
            winner = "White" if result == "1-0" else "Black"
            msg = f"Checkmate. {winner} wins!\nModel updated from this game."
        else:
            msg = f"Game over: {result}\nModel updated from this game."

        self.status.config(text=msg)
        messagebox.showinfo("Game Over", msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()
