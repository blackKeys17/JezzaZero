import chess
import numpy as np

from helpers import random_policy, material_eval, pesto_eval

class MCTSNode():
    def __init__(self, parent, cur_player, prev_move, c):
        self.parent = parent
        self.cur_player = cur_player
        self.c = c

        self.total_value = 0
        self.total_visits = 0

        # Stored as {move: node}
        self.children = {} 

        self.prev_move = prev_move
        self.expanded = False
        self.fully_expanded = False
        self.is_terminal = False
        self.untried_moves = []

    def __str__(self):
        return f"MCTS Node\nTotal value: {self.total_value}\nTotal visits: {self.total_visits}"
    
    # Returns index of maximum element
    def argmax(self, a: dict):
        return max(a.keys(), key=lambda x: a[x])
    
    # UCT formula for child nodes
    def UCT(self, N, Q):
        if self.cur_player == True:
            return Q/N + self.c * np.sqrt(np.log(self.total_visits) / N)
        else:
            return -(Q/N) + self.c * np.sqrt(np.log(self.total_visits) / N)

    # Selection of child node to traverse to using UCT formula
    def select(self):
        return self.argmax({node: self.UCT(node.total_visits, node.total_value) for node in self.children.values()})

    # Expansion - creates and returns a new child node
    def expand(self):
        move = self.untried_moves.pop()
        new_node = MCTSNode(self, not self.cur_player, move, self.c)
        self.children.update({move: new_node})

        if len(self.untried_moves) == 0:
            self.fully_expanded = True

        return new_node

    # Simulate game with random moves
    def rollout(self, board: chess.Board, rollout_depth):
        for i in range(rollout_depth*2):
            if board.is_game_over():
                break
            board.push_san(random_policy(board))
        
        return pesto_eval(board)
    
    # Backpropagate results
    def backpropagate(self, eval):
        self.total_value += eval
        self.total_visits += 1

        if self.parent != None:
            self.parent.backpropagate(eval)