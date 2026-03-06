"""
Evaluate a trained model against Stockfish at multiple skill levels to estimate ELO.

Usage:
    python eval_vs_stockfish.py --model models/chess_model_pretrained.pth
    python eval_vs_stockfish.py --model models/chess_model_500.pth --games 20 --stockfish /path/to/stockfish

Stockfish skill levels map roughly to ELO:
    0  ~ 800,  5  ~ 1100, 10 ~ 1500, 15 ~ 1900, 20 ~ 2850
"""

import argparse
import os
import random
import math

import chess
import chess.engine
import numpy as np
import torch

import chessBot0 as cb
import chessFunctions as cF

# Approximate ELO for each Stockfish skill level (0-20)
SKILL_ELO = {
    0:  800,
    2:  950,
    4:  1100,
    6:  1300,
    8:  1500,
    10: 1700,
    12: 1900,
    14: 2100,
    16: 2300,
    18: 2600,
    20: 2850,
}

MOVE_LIMIT = 150  # draw if game exceeds this many half-moves


def pick_move(board, net, device, depth=1, epsilon=0.0):
    if random.random() < epsilon:
        return random.choice(list(board.legal_moves))

    best_move = None
    best_val = float('-inf')

    for move in board.legal_moves:
        board.push(move)
        val = -_minimax(board, depth - 1, net, device)
        board.pop()
        if val > best_val:
            best_val = val
            best_move = move

    return best_move


def _minimax(board, depth, net, device):
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            return 0
        return 1 if outcome.winner == board.turn else -1

    if depth == 0:
        t = torch.tensor(cF.board_to_tensor(board)).unsqueeze(0).to(device)
        with torch.no_grad():
            return net(t).item()

    best = float('-inf')
    for move in board.legal_moves:
        board.push(move)
        val = -_minimax(board, depth - 1, net, device)
        board.pop()
        if val > best:
            best = val
    return best


def play_game(net, device, engine, skill_level, model_plays_white, depth=1):
    """Play one game. Returns 1=model win, 0=draw, -1=model loss."""
    engine.configure({"Skill Level": skill_level})
    board = chess.Board()

    while not board.is_game_over():
        if board.fullmove_number * 2 > MOVE_LIMIT:
            return 0  # draw by move limit

        model_turn = (board.turn == chess.WHITE) == model_plays_white

        if model_turn:
            move = pick_move(board, net, device, depth=depth)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.05))
            move = result.move

        board.push(move)

    outcome = board.outcome()
    if outcome.winner is None:
        return 0
    model_color = chess.WHITE if model_plays_white else chess.BLACK
    return 1 if outcome.winner == model_color else -1


def estimate_elo(results_by_skill):
    """
    Use the Elo expected score formula to estimate the model's ELO.
    E(score) = 1 / (1 + 10^((opponent_elo - model_elo) / 400))
    Fit model_elo to minimise squared error over all skill levels played.
    """
    from scipy.optimize import minimize_scalar

    def total_error(model_elo):
        err = 0.0
        for skill, (wins, draws, losses) in results_by_skill.items():
            n = wins + draws + losses
            if n == 0:
                continue
            actual_score = (wins + 0.5 * draws) / n
            opp_elo = SKILL_ELO[skill]
            expected = 1 / (1 + 10 ** ((opp_elo - model_elo) / 400))
            err += (actual_score - expected) ** 2
        return err

    res = minimize_scalar(total_error, bounds=(100, 3000), method='bounded')
    return round(res.x)


def run_eval(model_path, stockfish_path, games_per_level, skills, depth):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = cb.RecConNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()
    print(f"Loaded {model_path} on {device}")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    results_by_skill = {}
    for skill in skills:
        wins = draws = losses = 0
        for g in range(games_per_level):
            model_white = (g % 2 == 0)
            r = play_game(net, device, engine, skill, model_white, depth=depth)
            if r == 1:
                wins += 1
            elif r == 0:
                draws += 1
            else:
                losses += 1
            print(f"  skill {skill:2d} game {g+1}/{games_per_level}: {'W' if r==1 else 'D' if r==0 else 'L'}", flush=True)

        results_by_skill[skill] = (wins, draws, losses)
        score = (wins + 0.5 * draws) / games_per_level
        print(f"Skill {skill:2d} (~{SKILL_ELO[skill]} ELO): {wins}W {draws}D {losses}L  score={score:.2f}\n")

    engine.quit()

    try:
        elo = estimate_elo(results_by_skill)
        print(f"Estimated model ELO: ~{elo}")
    except Exception as e:
        print(f"Could not estimate ELO (scipy needed): {e}")
        print("Install with: pip install scipy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/chess_model_700.pth',
                        help='Path to .pth model file')
    parser.add_argument('--stockfish', default='stockfish',
                        help='Path to Stockfish binary (default: "stockfish" on PATH)')
    parser.add_argument('--games', type=int, default=10,
                        help='Games per skill level (default: 10, split evenly white/black)')
    parser.add_argument('--skills', type=int, nargs='+',
                        default=[0, 4, 8, 12],
                        help='Stockfish skill levels to test (default: 0 4 8 12)')
    parser.add_argument('--depth', type=int, default=1,
                        help='Minimax depth for model (default: 1)')
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model) \
        if not os.path.isabs(args.model) else args.model

    run_eval(model_path, args.stockfish, args.games, args.skills, args.depth)
