import chess
import json
import torch
from torch.utils.data import Dataset

from features import Features

# Reads in from a jsonl file
class FastChessDataset(Dataset):
    def __init__(self, file_path, num_games=0, start_game=0):
        f = open(file_path, "r")
        # Helpers
        features = Features()
        board = chess.Board()

        self.data = []
        for game_num, game in enumerate(f, 1):
            if game_num > num_games + start_game:
                break
            if game_num < start_game:
                continue

            game_data = json.loads(game)
            board.reset()

            winner = game_data["winner"]
            if winner == chess.WHITE:
                score = torch.tensor(0.8, dtype=torch.float32)
            elif winner == chess.BLACK:
                score = torch.tensor(-0.8, dtype=torch.float32)
            else:
                score = torch.tensor(0, dtype=torch.float32)
            
            position_tensors = [torch.zeros([8, 8, 12]) for i in range(3)]
            for move in game_data["moves"]:
                encoded_position = features.encode_pieces(board)
                position_tensors.append(encoded_position)
                net_in = features.encode_board(board, 4, position_tensors)

                if board.turn == chess.BLACK:
                    net_in = features.reflect_board(net_in, 4)
                
                # Reflect move for training
                if board.turn == chess.BLACK:
                    reflected_move = features.reflect_move_uci(move)

                # Add in this move to train from the previous position
                if board.turn == chess.WHITE:
                    x, y = int(move[1])-1, ord(move[0])-97
                elif board.turn == chess.BLACK:
                    x, y = int(reflected_move[1])-1, ord(reflected_move[0])-97

                # One hot encoding for training example
                encoded_next_move = torch.zeros([8, 8, 64], dtype=torch.float32)
                if board.turn == chess.WHITE:
                    move_type = features.move_type(move)
                elif board.turn == chess.BLACK:
                    move_type = features.move_type(reflected_move)
                encoded_next_move[x][y][move_type] = 1

                # Add training example as a triple: [board, move, score]
                self.data.append([net_in, encoded_next_move, score])

                board.push_uci(move)
            
        print("Finished loading in dataset :)\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

if __name__ == "__main__":
    x = FastChessDataset("alpha-mcts/net/training_data/lichess_elite_2022-02.jsonl", 200, 0)
    print(x[1][0][:, :, 24])
    print(len(x))
