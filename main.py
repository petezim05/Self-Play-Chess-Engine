import numpy as np
import chessBot0 as cb
import torch
import chess
import chessFunctions as cF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = cb.RecConNet().to(device)
_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chess_model_pretrained.pth')
net.load_state_dict(torch.load(_model_path, map_location=device, weights_only=True))
print("Loaded existing model weights.")


def minimax(board, depth, net):
    board = board
    
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            return 0
        return 1 if outcome.winner == board.turn else -1
    
    if depth == 0:
        tens = torch.tensor(cF.board_to_tensor(board)).unsqueeze(0).to(device)
        with torch.no_grad():
            return net(tens).item()
    
    bestVal = float('-inf')

    for move in board.legal_moves:
        board.push(move)
        val = -minimax(board , depth - 1 , net)
        board.pop()
        if val > bestVal:
            bestVal = val 
    return bestVal
    
def pickMove(board, depth, net, epsilon):

    if random.random() < epsilon:
        return random.choice(list(board.legal_moves))

    bestMove = None
    bestVal = float('-inf')
    board = board

    for move in board.legal_moves:
        board.push(move)
        val = -minimax(board , depth -1, net=net)
        board.pop()

        if val > bestVal:
            bestVal = val
            bestMove = move

    return bestMove

def playGame(net, depth , epsilon):
    board = chess.Board()
    positions = []

    while not board.is_game_over():
        moveLimit = 100
        if len(positions) >= moveLimit:
            return positions, 0  # treat as draw
        
        positions.append((cF.board_to_tensor(board), board.turn))
        board.push(pickMove(board, depth, net, epsilon))
        
    
    outcome = board.outcome()
    
    if outcome.winner == chess.WHITE: reward = 1
    if outcome.winner == chess.BLACK: reward = -1
    if outcome.winner is None: reward = 0

    return positions, reward
    

loss_fn = nn.MSELoss()
teacher = torch.optim.Adam(params= net.parameters(), lr=0.001)


def train(games, net, depth, batch):
    tensors = []
    labels = []
    b = 0

    eps = 1.0
    epsMin = .05
    epsDec = .995

    for i in range(games):
        positions, reward = playGame(net= net , depth= depth, epsilon=eps)
        eps = max(epsMin , eps*epsDec)
        for p , t in positions:
            tensors.append(p)
            if t == chess.WHITE:
                labels.append(reward)
            else:
                labels.append(-reward)

        batchLoss = None
        b += 1
        if b == batch:
            b = 0
            ts = torch.tensor(np.array(tensors)).to(device)
            labs = torch.tensor(labels , dtype=torch.float32).unsqueeze(1).to(device)

            tensors = []
            labels = []

            teacher.zero_grad()
            output = loss_fn(net(ts) , labs)
            batchLoss = output.item()
            output.backward()
            teacher.step()

        if i % batch == 0 and i >= batch:
            torch.save(net.state_dict(), f'models/chess_model_{i}.pth')
            print(f"game {i}, loss: {batchLoss}")

games = int(input("enter number of games: "))
batches = int(input("enter number of batches"))

train(games=games, net=net, depth=1, batch=batches)
