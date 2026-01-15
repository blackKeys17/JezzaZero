import chess
from chess.polyglot import zobrist_hash
import torch
import json
from collections import defaultdict

# DFS using the "tree", softer targets to use for training
def write_data(file, position_data, board: chess.Board, move_stack):
    # Base case
    if sum(position_data[zobrist_hash(board)][1]) <= 60:
        return

    # Process current position, into the form:
    # [Encoded board (with move history), move distribution, WDL distribution]
    move_stack.append(board.fen())
    temp = [None, None, None]
    temp[0] = move_stack[-4:]
    
    # TODO - Rescale to sum to 1
    zobrist = zobrist_hash(board)
    temp[1] = position_data[zobrist][0]
    temp[2] = position_data[zobrist][1]

    temp = json.dumps(temp)
    file.write(temp)
    file.write("\n")
    
    # Traverse down tree
    for move in board.generate_legal_moves():
        board.push(move)
        visited[zobrist] = True
        if not visited[zobrist_hash(board)]:
            write_data(file, position_data, board, move_stack)
        visited[zobrist_hash(board)] = False
        board.pop()
    
    return

input("Write game tree?")
total_games = 1000000
max_depth = 72
f = open("alpha-mcts/net/training_data/lichess_elite_2022-02.jsonl", "r")
position_data = defaultdict(lambda: [{}, [0, 0, 0]])
board = chess.Board()

for game_num, game in enumerate(f, 1):
    if game_num > total_games:
        break
    if game_num % 5000 == 0:
        print(f"Processed {game_num} games...")

    game_data = json.loads(game)
    board.reset()
    for halfmove_count, move in enumerate(game_data["moves"], 1):
        if halfmove_count > max_depth:
            break
        zobrist = zobrist_hash(board)
        # Move frequencies, WDL count
        position_data[zobrist][0].update({move: position_data[zobrist][0].get(move, 1) + 1})
        
        if game_data["winner"] == chess.WHITE:
            position_data[zobrist][1][0] += 1
        elif game_data["winner"] == chess.BLACK:
            position_data[zobrist][1][2] += 1
        else:
            position_data[zobrist][1][1] += 1
        
        board.push_uci(move)

f.close()
board.reset()
move_stack = [None for _ in range(3)]
visited = defaultdict(lambda: False)

f = open("alpha-mcts/net/training_data/lichess_elite_2022_soft_targets.jsonl", "w")
f.write("")
f.close()

f = open("alpha-mcts/net/training_data/lichess_elite_2022_soft_targets.jsonl", "a")
write_data(f, position_data, board, move_stack)
f.close()

print("Finished processing games")