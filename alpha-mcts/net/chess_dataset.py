import chess.pgn
import torch
from torch.utils.data import Dataset

from features import Features

# For supervised training on games stored in a PGN file
class ChessDataset(Dataset):
    def __init__(self, file, games, start=0, skip_draws=False):
        pgn = open(file)
        self.data = []
        self.features = Features()

        for i in range(start):
            chess.pgn.skip_game(pgn)
        
        count = 0        
        while True and count < games:
            if count % 10 == 0:
                print(count)
            game = chess.pgn.read_game(pgn)
            if game == None:
                break
            
            board = game.board()
            winner = game.headers["Result"]
            if winner == "1-0":
                score = torch.tensor(1, dtype=torch.float32)
            elif winner == "0-1":
                score = torch.tensor(-1, dtype=torch.float32)
            else:
                score = torch.tensor(0, dtype=torch.float32)
                if skip_draws:
                    continue
            
            for move in game.mainline_moves():
                if not board.is_game_over():
                    encoded_position = self.features.encode_board(board, 4)
                    
                # Add in this move to train from the previous position
                x, y = divmod(move.from_square, 8)

                # One hot encoding for training example
                encoded_next_move = torch.zeros([8, 8, 64], dtype=torch.float32)
                encoded_next_move[x][y][self.features.move_type(move.uci())] = 1

                # Add training example as a triple: [board, move, score]
                self.data.append([encoded_position, encoded_next_move, score])

                board.push(move)
            
            count += 1
        
        print("Loaded in games")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

if __name__ == "__main__":
    x = ChessDataset("alpha-mcts/net/training_data/lichess_elite_2022-02.pgn", 500)
    print(len(x))
    for i in range(3):
        print(x[i][0][:, :, 1])