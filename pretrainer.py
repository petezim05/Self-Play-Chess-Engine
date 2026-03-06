import numpy as np
import chessBot0 as cb
import torch
import chess
import chess.pgn
import chessFunctions as cF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = cb.RecConNet().to(device)

import os
#if os.path.exists('chess_model_pretrained.pth'):
#    net.load_state_dict(torch.load('chess_model_pretrained.pth', map_location=device, weights_only=True))
#    print("Loaded existing model weights.")



loss_fn = nn.MSELoss()
teacher = torch.optim.Adam(params= net.parameters(), lr=0.001)

def pretrain_from_pgn(net, pgn_path):
    game_num = 0
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_num += 1

            result = game.headers.get("Result", "*")
            if result == "1-0":
                winner = chess.WHITE
            elif result == "0-1":
                winner = chess.BLACK
            else:
                print(f"Game {game_num}: draw or result unknown, skipping.")
                continue

            board = game.board()
            tensors = []
            labels = []

            for move in game.mainline_moves():
                tensors.append(cF.board_to_tensor(board))
                labels.append(1.0 if board.turn == winner else 0.0)
                board.push(move)

            if not tensors:
                print(f"Game {game_num}: no positions collected, skipping.")
                continue

            ts = torch.tensor(np.array(tensors)).to(device)
            labs = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

            teacher.zero_grad()
            preds = net(ts)
            loss = loss_fn(preds, labs)
            loss_val = loss.item()
            loss.backward()
            teacher.step()

            torch.save(net.state_dict(), 'chess_model_pretrained.pth')
            print(f"Game {game_num}: trained on {len(tensors)} positions, loss: {loss_val:.4f}, model saved.")

    print(f"Done. Processed {game_num} games.")

pretrain_from_pgn(net=net, pgn_path="pgns.txt")
