import chess
import torch

from MCTS import MCTS_train, MCTSNode
from net.resnet import ResNet
from net.features import Features

net = ResNet(55, 128, 8, 64)
net.load_state_dict(torch.load("alpha-mcts/net/temp.pth"))
net.eval()
board = chess.Board()
root = MCTSNode(None, None, True, 1)
features = Features()

player_turn = True
while not board.is_game_over():
    if player_turn:
        move = MCTS_train(root, board, features, net, 2, 0.5, 1.25, 0.3, 0)
        print(f"Nodes evaluated: {root.visits}")
        print(f"Move: {root.children_moves[move]}")
        board.push_san(root.children_moves[move])
        print(board)
        print()
        #print("\n".join([f"{i}: {j} visits" for i,j in zip(root.children_moves, root.children_visits)]))
        root = root.children_nodes[move]
        root.parent = None

    else:
        valid_move = False
        while not valid_move:
            try:
                move = input("Enter move: ")
                board.push_san(move)
                print(board)
                root = root.children_nodes[root.children_moves.index(move)]
                valid_move = True
            except:
                print("Enter a valid move")
    
    player_turn = not player_turn