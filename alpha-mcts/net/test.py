import numpy as np
import torch
from torch.utils.data import DataLoader

from copy import deepcopy
from features import Features

# Test against top k moves
def test(net, dataset, k):
    f = Features()
    data_loader = DataLoader(dataset, batch_size=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_correct = 0
    num_topk_correct = 0
    num_samples = 0

    temp = deepcopy(net)
    temp.eval()

    with torch.no_grad():
        for board, moves, score in data_loader:
            board = board.to(device)
            net_out, _ = net(board)
            
            # For comparison
            net_out = torch.flatten(net_out, 1)
            moves = torch.flatten(moves, 1)
        
            # Get indices of moves with highest assigned probabilities
            net_top_idx = torch.argmax(net_out, 1)
            ground_top_idx = torch.argmax(moves, 1)
            net_top_k = torch.topk(net_out, k , 1)
            ground_top_k = torch.topk(moves, k , 1)

            for i in range(len(net_top_idx)):
                if net_top_idx[i] == ground_top_idx[i]:
                    num_correct += 1

            for i in range(len(net_top_k)):
                num_topk_correct += len(np.intersect1d(net_top_k[i].cpu(), ground_top_k[i].cpu()))

            num_samples += net_out.size(0)

    # [Proportion of correct top moves, average # of top-k moves]
    return [num_correct/num_samples, num_topk_correct/num_samples]