# chess0

A chess AI trained via reinforcement learning and self-play. A convolutional neural network learns to evaluate board positions, guided by minimax search, with game outcomes as the training signal.

## How it works

- The bot plays games against itself using **epsilon-greedy exploration** — starting with mostly random moves and gradually shifting toward learned strategy
- Each position is evaluated by a **neural network** (`RecConNet`) that outputs a score in [-1, 1]
- **Minimax search** (negamax) uses the network's evaluations to look ahead and pick moves
- After each batch of games, the network is trained on the recorded positions and their outcomes (win/loss/draw)

## Architecture

The board is encoded as an **18-channel 8×8 tensor**:
- Channels 0–11: piece locations (6 piece types × 2 colors)
- Channel 12: side to move
- Channels 13–16: castling rights
- Channel 17: en passant file

The network (`chessBot0.py`) uses convolutional layers for spatial pattern recognition followed by fully connected layers, with a tanh output.

## Project structure

```
chess0/
├── main.py              # Training loop entry point
├── chessBot0.py         # Neural network definition (RecConNet)
├── chessFunctions.py    # Board encoding utilities
├── chessSandbox.py      # Scratch/test file
└── chess_model_0.pth    # Saved model checkpoint
```

## Requirements

```
torch
python-chess
numpy
```

Install with:

```bash
pip install torch python-chess numpy
```

GPU acceleration is used automatically if CUDA is available.

## Usage

Run the training loop:

```bash
python main.py
```

Key parameters (in `main.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `games`   | 100     | Number of self-play training games |
| `depth`   | 1       | Minimax lookahead depth |
| `batch`   | 50      | Games per training batch |

Model checkpoints are saved every 50 games as `chess_model_*.pth`.
