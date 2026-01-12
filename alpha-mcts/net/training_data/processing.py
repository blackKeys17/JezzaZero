import chess.pgn
import json

# Just in case
input("Process training data?")

# Clear first
file = open("alpha-mcts/net/training_data/lichess_elite_2022-02.jsonl", "w")
file.write("")
file.close()

pgn = open("alpha-mcts/net/training_data/lichess_elite_2022-02.pgn")
file = open("alpha-mcts/net/training_data/lichess_elite_2022-02.jsonl", "a")

processed = 0
while True:
    game = chess.pgn.read_game(pgn)
    if game == None:
        break

    if processed % 10000 == 0:
        print(processed)

    game_data = {"moves": [], "winner": None}
    for move in game.mainline_moves():
        game_data["moves"].append(str(move))
    
    winner = game.headers["Result"]
    if winner == "1-0":
        game_data["winner"] = True
    elif winner == "0-1":
        game_data["winner"] = False
    else:
        game_data["winner"] = None
    
    game_data = json.dumps(game_data)
    file.write(game_data)
    file.write("\n")
    processed += 1

file.close()
    