import chess
import chess.pgn
from chess.polyglot import zobrist_hash

from copy import deepcopy
import json
from collections import defaultdict

# DFS using the "tree", softer targets to use for training
def write_data(file, position_data, board: chess.Board, move_stack, depth=1):
    # Base case
    if sum(position_data[zobrist_hash(board)][1]) <= 40:
        return

    # Process current position, into the form:
    # [Past board FENs, move distribution, WDL distribution]
    move_stack.append(board.fen())
    zobrist = zobrist_hash(board)

    if depth >= 10:
        temp = [None, None, None]
        temp[0] = move_stack[-4:]
        
        total_moves = sum(position_data[zobrist][1])
        
        temp[1] = deepcopy(position_data[zobrist][0])
        for key in temp[1].keys():
            temp[1][key] /= total_moves    
        temp[2] = deepcopy(position_data[zobrist][1])
        for i in range(len(temp[2])):
            temp[2][i] /= total_moves

        temp = json.dumps(temp)
        file.write(temp)
        file.write("\n")
    
    # Traverse down tree
    for move in board.generate_legal_moves():
        board.push(move)
        visited[zobrist] = True
        if not visited[zobrist_hash(board)]:
            write_data(file, position_data, board, move_stack, depth+1)
        visited[zobrist_hash(board)] = False
        board.pop()
    
    return

input("Write game tree?")
total_games = 100000000
max_depth = 100
pgn = open("alpha-mcts/net/training_data/lichess_elite_2023-11.pgn")
position_data = defaultdict(lambda: [{}, [0, 0, 0]])
board = chess.Board()
game_num = 0

while True:
    if game_num > total_games:
        break
    if (game_num) % 5000 == 0:
        print(f"Processed {game_num} games...")

    game_data = chess.pgn.read_game(pgn)
    if game_data == None:
        break
    board.reset()

    for move in game_data.mainline_moves():
        if board.halfmove_clock > max_depth:
            break
        zobrist = zobrist_hash(board)

        # Move frequencies, WDL count
        position_data[zobrist][0].update({move.uci(): position_data[zobrist][0].get(move, 0) + 1})
        
        if game_data.headers["Result"] == "1-0":
            position_data[zobrist][1][0] += 1
        elif game_data.headers["Result"] == "0-1":
            position_data[zobrist][1][2] += 1
        else:
            position_data[zobrist][1][1] += 1
        
        board.push(move)
    
    game_num += 1

pgn.close()
board.reset()
move_stack = [None for _ in range(3)]
visited = defaultdict(lambda: False)

f = open("alpha-mcts/net/training_data/lichess_elite_2023_11_soft_targets.jsonl", "w")
f.write("")
f.close()

f = open("alpha-mcts/net/training_data/lichess_elite_2023_11_soft_targets.jsonl", "a")
write_data(f, position_data, board, move_stack)
f.close()

print("Finished processing games")