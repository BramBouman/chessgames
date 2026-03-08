import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Transformer-based chess player using a causal LM.
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "TinyLMPlayer",
        model_id: str = "BramBouman/chess_llm_135M",
        temperature: float = 0.01,
        max_new_tokens: int = 4,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.board = chess.Board()
        self.move_history = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded components
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # -------------------------
    # Update board and move history
    # -------------------------
    def _update_board(self, fen: str):
        # New game detection
        if fen.startswith("rnbqkbnr/pppppppp/"):
            self.board = chess.Board()
            self.move_history = []

        # No change
        if self.board.fen() == fen:
            return

        # Try to find the opponent move
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.fen() == fen:
                self.move_history.append(move.uci())
                return
            self.board.pop()

        # Fallback: reset board
        self.board = chess.Board(fen)
        self.move_history = []

    # -------------------------
    # Build prompt from move history
    # -------------------------
    def _build_prompt(self):
        history = self.move_history[-30:]
        turn = "W:" if self.board.turn == chess.WHITE else "B:"

        if not history:
            return f"{turn} START\n"

        return f"{turn} " + " ".join(history) + "\n"

    # -------------------------
    # Choose the best legal move
    # -------------------------
    def _choose_move(self, fen: str) -> Optional[str]:
        self._load_model()
        board = self.board
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None

        prompt = self._build_prompt()
        sequences = [prompt + move for move in legal_moves]

        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**batch)

        logits = outputs.logits[:, -1, :]

        # Map each move to its token id (last token)
        move_ids = []
        for move in legal_moves:
            tokens = self.tokenizer(move, add_special_tokens=False)["input_ids"]
            move_ids.append(tokens[-1])

        scores = [logits[i, move_id].item() for i, move_id in enumerate(move_ids)]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return legal_moves[best_idx]

    # -------------------------
    # Random legal fallback
    # -------------------------
    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # -------------------------
    #  API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._update_board(fen)
            move = self._choose_move(fen)

            if move:
                self.board.push(chess.Move.from_uci(move))
                self.move_history.append(move)
                return move

        except Exception as e:
            print(f"[{self.name}] Exception in get_move: {e}")

        # fallback
        return self._random_legal(fen)

