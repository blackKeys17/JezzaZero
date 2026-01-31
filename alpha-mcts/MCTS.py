import chess
import copy
import torch

from net.features import Features

# TODO - rewrite to use new bitboard encoding (which mirrors vertically)
class MCTSNode():
    def __init__(self, parent, move, player_turn, c_puct):
        self.parent = parent
        self.children_moves = None # List
        self.children_nodes = None # List
        self.children_total_action_value = None # Torch tensor
        self.children_visits = None # Torch tensor
        self.children_masked_priors = None # Torch tensor
        self.last_expanded = None
        self.is_expanded = False

        self.move = move
        self.player_turn = player_turn
        self.is_terminal = False

        # Values for PUCT
        self.predicted_value = 0 # Evaluation from value head
        self.total_action_value = 0 # MCTS total evaluation from backprop
        self.visits = 0
        self.c_puct = c_puct
    
    def __str__(self):
        return f"MCTS Node\nMove: {self.move}\nAverage value: {self.total_value / self.visits}\nVisits: {self.visits}\n"
    
    # Probabilistic upper confidence bound for trees formula on all children
    def puct(self):
        visit_mask = self.children_visits > 0
        q_vals = torch.zeros_like(self.children_visits)
        q_vals[visit_mask] = self.children_total_action_value[visit_mask] / self.children_visits[visit_mask]
        
        # Approximate Q-values for remaining children using parent's Q-value
        q_vals[~visit_mask] = -self.total_action_value / self.visits

        # Sum approximated Q-values and policy head probabilities
        return q_vals + self.c_puct * self.children_masked_priors * torch.sqrt(self.visits / (1 + self.children_visits))
    
    # Returns index of puct-maximising node for selection stage of MCTS
    def select(self):
        return torch.argmax(self.puct())

# MCTS from a given root node and position, for training (subtrees are reused so tree might already be partially constructed)
# Returns the index of the next move to be played
def MCTS_train(root: MCTSNode, board: chess.Board, features: Features, net, simulations, temp, c_puct, alpha, epsilon):
    # Reset parent so values/visits aren't backpropagated further up the tree
    root.parent = None

    for _ in range(simulations):
        cur_pos = copy.deepcopy(board)

        # Traverse down tree to node to be expanded
        cur_node = root
        while cur_node.is_expanded:
            move_index = cur_node.select()
            cur_node.last_expanded = move_index
            cur_pos.push_san(cur_node.children_moves[move_index])
            cur_node = cur_node.children_nodes[move_index]
        
        # Update this node with all of its relevant information
        cur_node.children_moves = [str(move) for move in cur_pos.legal_moves]
        cur_node.children_nodes = [MCTSNode(cur_node, move, not cur_node.player_turn, c_puct) for move in cur_node.children_moves]
        cur_node.children_total_action_value = torch.zeros([len(cur_node.children_moves)])
        cur_node.children_visits = torch.zeros([len(cur_node.children_moves)])
        cur_node.is_expanded = True

        # Using the policy head, adding Dirichlet noise to priors during training (controlled by alpha and epsilon)
        policy, value = net(torch.unsqueeze(features.encode_board(cur_pos, 4), 0))
        policy, value = policy[0], torch.softmax(value[0], dim=0)
        # print(f"please {torch.sum(policy)}")
        # print(f"PLEASE {torch.sum(value)}")
        # print("\n".join([f"{str(i)}: {str(j.item())}" for i,j in list(zip(cur_node.children_moves, features.get_move_priors(features.mask_illegal(policy, cur_node.children_moves, board.turn), cur_node.children_moves, board.turn)))]))
        # print(f"Node evaluation: {value[0] - value[2]}")
        cur_node.children_masked_priors = features.add_dirichlet_noise(features.get_move_priors(features.mask_illegal(policy, cur_node.children_moves, board.turn), cur_node.children_moves, board.turn), alpha, epsilon)

        # Record the evaluation of the current node by the value head for training later, then backpropagate value
        value_score = value[0] - value[2]
        cur_node.predicted_value = value_score
        cur_node.total_action_value += value_score
        cur_node.visits += 1
        while cur_node.parent != None:
            cur_node = cur_node.parent
            cur_node.total_action_value += value_score
            cur_node.visits += 1
             
            # Update stats of stored children
            value_score = -value_score
            cur_node.children_total_action_value[cur_node.last_expanded] += value_score
            cur_node.children_visits[cur_node.last_expanded] += 1
        
    # TODO - handle reflection here
    # Select move to be played from visit count
    next_move_probs = features.next_move_dist(root.children_visits, temp)
    next_move = torch.multinomial(next_move_probs, 1)

    return next_move