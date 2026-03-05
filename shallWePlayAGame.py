import torch
import chess
import chessBot0 as cb
import chessFunctions as cF
from main import pickMove

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = cb.RecConNet().to(device)
net.load_state_dict(torch.load('chess_model_pretrained.pth', map_location=device))
net.eval()

board = chess.Board()

while not board.is_game_over():
    print(board)
    print()
    
    if board.turn == chess.WHITE:
        move = input("your move (e.g. e2e4): ")
        try:
            board.push_uci(move)
        except:
            print("illegal move, try again")
            continue
    else:
        move = pickMove(board, depth=1, net=net, epsilon=0)
        print(f"bot plays: {move}")
        board.push(move)

print(board)
print(board.outcome())