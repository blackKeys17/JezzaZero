import chess
import random
import math

# Rough heuristic for first attempt at MCTS
def material_eval(board: chess.Board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    eval = 0
    for _, piece in board.piece_map().items():
        # White piece
        if piece.color == True:
            eval += piece_values[piece.piece_type]
        # Black piece
        else:
            eval -= piece_values[piece.piece_type]
    
    return eval

# Pesto evaluation
import chess
import math

def pesto_eval(board: chess.Board):
    if board.is_game_over():
        if board.result() == True:
            return 1.0
        elif board.result == False:
            return -1.0
        else:
            return 0.0

    mg_val = {1: 82, 2: 337, 3: 365, 4: 477, 5: 1025, 6: 0}
    eg_val = {1: 94, 2: 281, 3: 297, 4: 512, 5: 936, 6: 0}
    phase_inc = [0, 0, 1, 1, 2, 4, 0]

    pst = {
        1: (
            [0,0,0,0,0,0,0,0, -35,-1,-20,-23,-15,24,38,-22, -26,-4,-4,-10,3,3,33,-12, -27,-2,-5,12,17,6,10,-25, -14,13,6,21,23,12,17,-23, -6,7,26,31,65,56,25,-20, 98,134,61,95,68,126,34,-11, 0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0, 13,8,8,10,13,0,2,-7, 4,7,-6,1,0,-5,-1,-8, 13,9,-3,-7,-7,-8,3,-1, 32,24,13,5,-2,4,17,17, 94,100,85,67,56,53,82,84, 178,173,158,134,147,132,165,187, 0,0,0,0,0,0,0,0]),
        2: (
            [-105,-21,-58,-33,-17,-28,-19,-23, -29,-53,-12,-3,-1,18,-14,-19, -23,-9,12,10,19,17,25,-16, -13,4,16,13,28,19,21,-8, -9,17,19,53,37,69,18,22, -47,60,37,65,84,129,73,44, -73,-41,72,36,23,62,7,-17, -167,-89,-34,-49,61,-97,-15,-107],
            [-60,-38,-22,-14,-22,-26,-24,-34, -8,-20,23,-10,-12,-33,-29,-38, -23,-3,-1,15,10,-3,-20,-22, -18,-6,16,25,16,17,4,-18, -17,3,22,22,22,11,8,-18, -24,-20,10,9,-1,-9,-19,-41, -25,-8,-25,-2,-9,-25,-24,-52, -58,-38,-13,-28,-31,-27,-63,-99]),
        3: (
            [-33,-3,-14,-21,-13,-12,-39,-21, 4,15,16,0,7,21,33,1, 0,15,15,15,14,27,18,10, -6,13,13,26,34,12,10,4, -4,5,19,50,37,37,7,-2, -16,37,43,40,35,50,37,-2, -26,16,-18,-13,30,59,18,-47, -29,4,-82,-37,-25,-42,7,-8],
            [-16,-21,-14,-10,-6,-6,-29,-15, -3,7,2,2,-6,-11,0,-4, -2,13,12,4,8,17,13,-1, -6,14,15,15,13,22,9,-4, -3,5,13,15,5,14,5,-1, -4,-8,8,20,37,-4,-1,-8, -8,-4,7,-12,-3,-13,-4,-14, -14,-21,-11,-8,-7,-9,-17,-24]),
        4: (
            [-19,-13,1,17,16,7,-37,-26, -44,-16,-20,-9,-1,11,-6,-71, -45,-25,-16,-17,3,0,-5,-33, -36,-26,-12,-1,9,-7,6,-23, -24,-11,7,26,24,35,-8,-20, -5,19,26,36,17,45,61,16, 27,32,58,62,80,67,26,44, 32,42,32,51,63,9,31,43],
            [-9,2,3,-1,-5,-13,4,-20, -6,-6,0,2,-9,-9,-11,-3, -4,0,-5,-1,-7,-12,-8,-16, 3,5,8,4,-5,-6,-8,-11, 4,3,13,1,2,1,-1,2, 7,7,7,5,4,-3,-5,-3, 11,13,13,11,-3,3,8,3, 13,10,18,15,12,12,8,5]),
        5: (
            [-1,-18,-9,10,-15,-25,-31,-50, -35,-8,11,2,8,15,-3,1, -14,2,-11,-2,-5,2,14,5, -9,-26,-9,-10,-2,-4,3,-3, -27,-27,-16,-16,-1,17,-2,1, -13,-17,7,8,29,56,47,57, -24,-39,-5,1,-16,57,28,54, -28,0,29,12,59,44,43,45],
            [-33,-28,-22,-43,-5,-32,-20,-41, -22,-23,-30,-16,-16,-23,-36,-32, -16,-27,15,6,9,17,10,5, -18,28,19,47,31,34,39,23, 3,22,24,45,57,40,57,36, -20,6,9,49,47,35,19,9, -17,20,32,41,58,25,30,0, -9,22,22,27,27,19,10,20]),
        6: (
            [-15,36,12,-54,8,-28,24,14, 1,7,-8,-64,-43,-16,9,8, -14,-14,-22,-46,-44,-30,-15,-27, -49,-1,-27,-39,-46,-44,-33,-51, -17,-20,-12,-27,-30,-25,-14,-36, -9,24,2,-16,-20,6,22,-22, 29,-1,-20,-7,-8,-4,-38,-29, -65,23,16,-15,-56,-34,2,13],
            [-53,-34,-21,-11,-28,-14,-24,-43, -27,-11,4,13,14,4,-5,-17, -19,-3,11,21,23,16,7,-9, -18,-4,21,24,27,23,9,-11, -8,22,24,27,26,33,26,3, 10,17,23,15,20,45,44,13, -12,17,14,17,17,38,23,11, -74,-35,-18,-18,-11,15,4,-17])
    }

    mg_score = 0
    eg_score = 0
    game_phase = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            ptype = piece.piece_type
            color = piece.color
            
            idx = chess.square_mirror(square) if color == chess.WHITE else square
            mg_p, eg_p = pst[ptype]
            
            m_val = mg_val[ptype] + mg_p[idx]
            e_val = eg_val[ptype] + eg_p[idx]

            if color == chess.WHITE:
                mg_score += m_val
                eg_score += e_val
            else:
                mg_score -= m_val
                eg_score -= e_val
            game_phase += phase_inc[ptype]

    mg_phase = min(game_phase, 24)
    eg_phase = 24 - mg_phase
    raw_score = (mg_score * mg_phase + eg_score * eg_phase) / 24
    norm_score = math.tanh(raw_score / 300.0)

    return norm_score

# For rollouts
def random_policy(board: chess.Board):
    num_moves = board.legal_moves.count()
    legal_moves_iter = iter(board.legal_moves)

    for i in range(random.randrange(num_moves)):
        next(legal_moves_iter)
    return str(next(legal_moves_iter))

if __name__ == "__main__":
    board = chess.Board()
    print(board)
    print(board.piece_map()[63].piece_type)
    
    print(pesto_eval(board))
    board.push_san("e2e4")
    print(pesto_eval(board))
    board.push_san("e7e5")
    board.push_san("g2g4")
    
    print(str(list(board.legal_moves)[0]))
    print(board)
    print(board.outcome().winner)