import chess
from copy import deepcopy
from MCTS import MCTSNode

def get_move(board: chess.Board):
    root = MCTSNode(None, board.turn, None, 0.6)
    
    for i in range(20000):
        temp_board = deepcopy(board)

        # Select
        cur_node = root
        while cur_node.fully_expanded:
            cur_node = cur_node.select()
            temp_board.push_san(cur_node.prev_move)
        
        # Expand
        if not cur_node.expanded:
            if temp_board.is_game_over():
                cur_node.is_terminal = True
            else:
                cur_node.untried_moves = list([str(i) for i in list(temp_board.legal_moves)])
            cur_node.expanded = True

        if not cur_node.is_terminal:
            new_node = cur_node.expand()
            eval = new_node.rollout(temp_board, 0)

        # Backpropagate
        new_node.backpropagate(eval)
    
    for i in root.children.values():
        print(i)
    return max(root.children.keys(), key=lambda x: root.children[x].total_visits)
        
board = chess.Board()
while True:
    move = get_move(board)
    board.push_san(move)
    print(move)
    print(board)
    print()
    valid = False
    while not valid:
        try:
            move = input("enter move: ")
            board.push_san(move)
            print(board)
            print()
            valid = True
        except:
            print("enter valid move")