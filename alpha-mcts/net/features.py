import chess
import copy
import torch
import torch.nn.functional as F

# Functions for converting between boards and arrays for the NN
class Features():
    def __init__(self):
        # Which channel each piece gets represented by
        self.piece_encodings = {
            "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
            "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
        }
    
    # Only encodes pieces into a tensor for quicker construction during training
    def encode_pieces(self, board: chess.Board):
        features = torch.zeros([8, 8, 12])
        for pos, piece in board.piece_map().items():
            x, y = divmod(pos, 8)
            features[x][y][self.piece_encodings[str(piece)]] = 1
        
        return features

    # TODO - HEAVILY optimise this using bitboards/anything else
    # Convert board object into a tensor of features, storing previous moves as well
    def encode_board(self, board: chess.Board, move_history, position_stack=None):
        temp = copy.deepcopy(board)
        features = torch.zeros([8, 8, move_history * 12 + 7])

        if position_stack != None:
            for i in range(move_history):
                features[:, :, i*12 : i*12+12] = position_stack[-1-i]

        else:
            game_start = False
            for i in range(move_history):
                # Treating a1 as [0][0], h1 as [0][7], a8 as [7][0] and h8 as [7][7]
                if not game_start:
                    for pos, piece in temp.piece_map().items():
                        x, y = divmod(pos, 8)
                        features[x][y][12*i + self.piece_encodings[str(piece)]] = 1
                # Keep as zeros if there aren't enough games
                else:
                    pass
                
                if len(temp.move_stack) == 0:
                    game_start = True
                else:
                    temp.pop()
        
        # Current player colour
        features[:, :, 12*move_history] = 1 if temp.turn == chess.WHITE else 0

        # Castling rights
        features[:, :, 12*move_history + 1] = temp.has_kingside_castling_rights(chess.WHITE)
        features[:, :, 12*move_history + 2] = temp.has_queenside_castling_rights(chess.WHITE)
        features[:, :, 12*move_history + 3] = temp.has_kingside_castling_rights(chess.BLACK)
        features[:, :, 12*move_history + 4] = temp.has_queenside_castling_rights(chess.BLACK)

        # Repeated moves
        features[:, :, 12*move_history + 5] = temp.halfmove_clock

        # Number of moves
        features[:, :, 12*move_history + 6] = temp.fullmove_number

        return features
    
    # TODO
    # Build features using list of FEN strings
    def encode_fen(self, fen):
        pass

    # Reflect the board representation for black (helps speed up training)
    def reflect_board(self, features, move_history):
        out = torch.zeros_like(features)

        # Swap white and black planes, flip vertically
        for i in range(move_history):
            out[:, :, i*12: i*12+6] = features[:, :, i*12+6: i*12+12]
            out[:, :, i*12+6: i*12+12] = features[:, :, i*12: i*12+6]
        out[:, :, 0:12*move_history] = torch.flip(out[:, :, 0:12*move_history], dims=[0])

        # Add in metadata planes
        out[:, :, 12*move_history: 12*move_history+7] = features[:, :, 12*move_history: 12*move_history+7]

        return out

    # Reflect a move from a network output
    def reflect_move_uci(self, move):
        out = ""
        out = f"{move[0]}{9-int(move[1])}{move[2]}{9-int(move[3])}"
        # Promotions
        if len(move) > 4:
            out += move[4]
        
        return out

    # Helper function for finding the direction of a move and mapping it to a corresponding output layer
    def move_type(self, move):
        # Change in x and y positions
        dx = ord(move[2]) - ord(move[0])
        dy = int(move[3]) - int(move[1])

        # Queen moves (going anticlockwise starting east)
        if dx > 0 and dy == 0: # E
            return dx - 1
        elif dx == dy and dx > 0: # NE
            return 7 + dx - 1
        elif dx == 0 and dy > 0: # N
            return 14 + dy - 1
        elif dx == -dy and dx < 0: # NW
            return 21 + abs(dx) - 1
        elif dx < 0 and dy == 0: # W
            return 28 + abs(dx) - 1
        elif dx == dy and dx < 0: # SW
            return 35 + abs(dx) - 1
        elif dx == 0 and dy < 0: # S
            return 42 + abs(dy) - 1
        elif dx == -dy and dx > 0: # SE
            return 49 + dx - 1
        
        # Knight moves (going anticlockwise starting north-east-east)
        elif dx == 2 and dy == 1:
            return 56
        elif dx == 1 and dy == 2:
            return 57
        elif dx == -1 and dy == 2:
            return 58
        elif dx == -2 and dy == 1:
            return 59
        elif dx == -2 and dy == -1:
            return 60
        elif dx == -1 and dy == -2:
            return 61
        elif dx == 1 and dy == -2:
            return 62
        elif dx == 2 and dy == -1:
            return 63

    # TODO
    # Generate training label from move distribution
    def soft_move_target(self, move_dist):
        pass

    # Exponential formula with temperature to get distribution to pick moves from 
    def next_move_dist(self, visits, temp):
        # Apply exponential formula
        total = sum([n ** (1/temp) for n in visits])
        policy_probs = torch.zeros([len(visits)])
        for i in range(len(visits)):
            policy_probs[i] = (visits[i] ** (1/temp)) / total

        return policy_probs

    # Gets distribution from visit count, where probability is proportional to visit count
    # Converts it to a tensor which is used in self-play learning
    # Note that this already masks out illegal moves - child nodes are only created using legal moves
    def policy_from_visits(self, moves, visits):
        total = sum(visits)
        policy_probs = []
        for move, n in zip(moves, visits):
            policy_probs.append([move, n/total])

        mcts_policy = torch.zeros([8, 8, 64])

        # First 2 dimensions represent squares on the board
        # First 56 channels in 3rd dimension represent queen moves
        # Last 8 represent knight moves
        for move, prob in policy_probs:
            # Get position of moving piece
            x, y = divmod(chess.parse_square(move[:2]), 8)
            mcts_policy[x][y][self.move_type(move)] = prob
        
        return mcts_policy
    
    # Mask out illegal moves from a policy head output, flipping for black
    def mask_illegal(self, policy, legal_moves, turn):
        # Fill with values acting like -ve infinity
        masked_policy = torch.full(policy.shape, -1e9)
        for move in legal_moves:
            if turn == chess.WHITE:
                x, y = divmod(chess.parse_square(move[:2]), 8)
                masked_policy[x][y][self.move_type(move)] = policy[x][y][self.move_type(move)]
            elif turn == chess.BLACK:
                x, y = divmod(chess.parse_square(self.reflect_move_uci(move)[:2]), 8)
                masked_policy[x][y][self.move_type(self.reflect_move_uci(move))] = policy[x][y][self.move_type(self.reflect_move_uci(move))]
        
        return masked_policy

    # Get masked policy output and softmax into a probability distribution
    def get_move_priors(self, policy, moves, turn):
        priors = torch.zeros([len(moves)])
        for i in range(len(moves)):
            if turn == chess.WHITE:
                x, y = divmod(chess.parse_square(moves[i][:2]), 8)
                priors[i] = policy[x][y][self.move_type(moves[i])]
            elif turn == chess.BLACK:
                x, y = divmod(chess.parse_square(self.reflect_move_uci(moves[i])[:2]), 8)
                priors[i] = policy[x][y][self.move_type(self.reflect_move_uci(moves[i]))]
        
        return F.softmax(priors, dim=0)
    
    # Add Dirichlet noise in place to priors (for diversifying move selection during training)
    # alpha is a concentration parameter (use 0.3 for chess), epsilon is an exploration parameter (can use araound 0.25)
    def add_dirichlet_noise(self, priors, alpha, epsilon):
        dirichlet = torch.distributions.Dirichlet(torch.full(priors.shape, alpha))
        return (1 - epsilon) * priors + epsilon * dirichlet.sample()

if __name__ == "__main__":
    x = Features()
    priors = torch.tensor([0.2, 0.3, 0.5])
    print(x.add_dirichlet_noise(priors, 0.3, 0.25))