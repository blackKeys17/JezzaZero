import chess
import json
import torch
from torch.utils.data import Dataset

from features import Features

class TreeDataset(Dataset):
    def __init__(self, file_path):
        f = open(file_path, "r")
        features = Features()
        
        self.data = []
        for position_data in f:
            fen, move_dist, wdl_dist = json.loads(position_data)
            self.data.append([features.encode_fen(fen), features.soft_move_target(move_dist), torch.tensor(wdl_dist)])
        
        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
x = TreeDataset("alpha-mcts/net/training_data/lichess_elite_2022_soft_targets.jsonl")
print(len(x))
for i in x[0]:
    print(i)