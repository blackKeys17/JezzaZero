import chess
import json
import torch
from torch.utils.data import Dataset

from features import Features

class TreeDataset(Dataset):
    def __init__(self, file_path, num_positions=float("inf")):
        f = open(file_path, "r")
        self.features = Features()
        
        self.data = []
        for i, position_data in enumerate(f, 1):
            if i > num_positions:
                break
            self.data.append(json.loads(position_data))
        
        f.close()
        print("Loaded in dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fen, move_dist, wdl_dist = self.data[index]
        if fen[-1].split()[1] == "w":
            return [self.features.encode_fen(fen), self.features.soft_move_target(move_dist), torch.tensor(wdl_dist)]
        else:
            return [self.features.encode_fen(fen), self.features.soft_move_target(move_dist), torch.tensor(wdl_dist[::-1])]
    
if __name__ == "__main__":
    x = TreeDataset("alpha-mcts/net/training_data/lichess_elite_2022_soft_targets.jsonl")
    print(len(x))
    for i in x[0]:
        print(i)