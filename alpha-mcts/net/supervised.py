import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from fast_chess_dataset import FastChessDataset
from resnet import ResNet

from matplotlib import pyplot as plt
import time

batch_size = 256
train_set = FastChessDataset("alpha-mcts/net/training_data/lichess_elite_2022-02.jsonl", 4000, 28000)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = ResNet(55, 128, 8, 64)
net.load_state_dict(torch.load("alpha-mcts/net/weights.pth"))
net.train()
net.to(device)

policy_criterion = nn.CrossEntropyLoss()
value_criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)

# Output loss every 400 batches
cur_loss = 0
cur_policy_loss = 0
cur_value_loss = 0
x_labels = []
plot_losses = []

torch.backends.cudnn.benchmark = True

# Start of main training loop
for epoch in range(2):
    start = time.perf_counter()
    cur_loss = 0
    cur_policy_loss = 0
    cur_value_loss = 0

    for i, data in enumerate(train_loader, 1):
        board, move, score = data[0].to(device), data[1].to(device), data[2].to(device)
        optimiser.zero_grad()

        policy, value = net(board)
        # Flatten all out to apply cross-entropy loss
        move = move.view(batch_size, -1)
        policy = policy.view(batch_size, -1)
        
        # Compute loss
        policy_loss = policy_criterion(policy, move) 
        value_loss = value_criterion(value, score)
        loss = policy_loss + 4 * value_loss
        
        cur_policy_loss += policy_loss.item()
        cur_value_loss += value_loss.item()
        cur_loss += loss.item()

        # Backpropagate and update weights
        loss.backward()
        # print(net.conv_pol.weight.grad.abs().sum())
        optimiser.step()
        
        if i%100 == 0:
            print(f"Epoch: {epoch + 1}, Batch number: {i}, Policy loss: {cur_policy_loss/100:.4f}, Value Loss: {cur_value_loss/100:.4f}")
            print(f"GPU temperature: {torch.cuda.temperature()} degrees")
            x_labels.append(epoch * len(train_set) + i)
            plot_losses.append(cur_loss/100)
            
            cur_loss = 0
            cur_policy_loss = 0
            cur_value_loss = 0
        
    epochTime = time.perf_counter() - start
    print(f"\nEpoch {epoch + 1} time: {epochTime:.4f}s\n")

input("Write new weights to file?")
print("New weights have been saved")
torch.save(net.state_dict(), "alpha-mcts/net/weights.pth")
plt.plot(x_labels, plot_losses)
plt.show()