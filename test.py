import numpy as np
import chessBot0 as cb
import torch
import chess
import chessFunctions as cF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = cb.RecConNet().to(device)


def minimax(board, depth, net):
    board = board
    
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            return 0
        return 1 if outcome.winner == board.turn else -1
    
    if depth == 0:
        tens = torch.tensor(cF.board_to_tensor(board)).unsqueeze(0).to(device)
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
        positions.append((cF.board_to_tensor(board), board.turn))
        board.push(pickMove(board, depth , net , epsilon))
        
    
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
        print(f"starting game {i}")  # is the loop running?
        
        positions, reward = playGame(net=net, depth=depth, epsilon=eps)
        print(f"game {i} done, positions: {len(positions)}, reward: {reward}")
        
        eps = max(epsMin, eps * epsDec)
        
        for p, t in positions:
            tensors.append(p)
            if t == chess.WHITE:
                labels.append(reward)
            else:
                labels.append(-reward)
        
        print(f"tensors: {len(tensors)}, b: {b}, batch: {batch}")  # is b incrementing?
        
        b += 1
        if b == batch:
            print("batch triggered")  # is this ever reached?
            b = 0
            ts = torch.tensor(np.array(tensors)).to(device)
            labs = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)
            tensors = []
            labels = []
            teacher.zero_grad()
            output = loss_fn(net(ts), labs)
            output.backward()
            teacher.step()
            print(f"loss: {output.item():.4f}")


train(games=5, net=net, depth=1, batch=5)
