import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from tree_dataset import TreeDataset
from resnet import ResNet
from test import test

# Logging
import time
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# TODO - Add validation loss and validation accuracy to training loop

batch_size = 512
train_set = TreeDataset("alpha-mcts/net/training_data/lichess_elite_2023_11_soft_targets.jsonl", 10000000)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = TreeDataset("alpha-mcts/net/training_data/lichess_elite_2022_soft_targets.jsonl", 102400)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet(55, 128, 8, 64)
# net.load_state_dict(torch.load("alpha-mcts/net/weights.pth"))
net.train()
net.to(device)

policy_criterion = nn.KLDivLoss(reduction="batchmean")
value_criterion = nn.KLDivLoss(reduction="batchmean")
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

# Output loss every 400 batches
cur_loss = 0
cur_policy_loss = 0
cur_value_loss = 0
x_labels = []
plot_losses = []

torch.backends.cudnn.benchmark = True

# Start of main training loop
for epoch in range(3):
    start = time.perf_counter()
    cur_loss = 0
    cur_policy_loss = 0
    cur_value_loss = 0

    for i, data in enumerate(train_loader, 1):
        board, move, score = data[0].to(device), data[1].to(device), data[2].to(device)
        optimiser.zero_grad()

        policy, value = net(board)
        # Flatten all out to apply cross-entropy loss
        move = torch.flatten(move, 1)
        policy = torch.flatten(policy, 1)
        
        # Move into logspace
        policy = F.log_softmax(policy, dim=1)
        value = F.log_softmax(value, dim=1)
        
        # Compute loss
        policy_loss = policy_criterion(policy, move) 
        value_loss = value_criterion(value, score)
        loss = policy_loss + value_loss
        
        policy_loss_item = policy_loss.item()
        cur_policy_loss += policy_loss_item
        value_loss_item = value_loss.item()
        cur_value_loss += value_loss_item
        loss_item = loss.item()
        cur_loss += loss_item

        # Backpropagate and update weights
        loss.backward()
        # print(net.conv_pol.weight.grad.abs().sum())
        optimiser.step()
        
        if i%50 == 0:
            print(f"Epoch: {epoch + 1}, Batch number: {i}, Policy loss: {cur_policy_loss/50:.5f}, Value Loss: {cur_value_loss/50:.5f}")
            print(f"GPU temperature: {torch.cuda.temperature()} degrees")
            cur_policy_loss = 0
            cur_value_loss = 0
            cur_loss = 0
        
        writer.add_scalar("Loss/train", loss_item, epoch * len(train_loader) + i)
        writer.add_scalar("Policy_Loss/train", policy_loss_item, epoch * len(train_loader) + i)
        writer.add_scalar("Value_Loss/train", value_loss_item, epoch * len(train_loader) + i)
        
        loss_item = 0
        policy_loss_item = 0
        value_loss_item = 0
        
    # TODO - Log training set and test set accuracy
    top, top_k = test(net, train_set, 3)
    print(f"Top move accuracy on training set: {top}")
    print(f"Top 3 move average on training set: {top_k}")
    writer.add_scalar("Training_accuracy/top_move_accuracy", top, epoch)
    writer.add_scalar("Training_accuracy/top_k_move_accuracy", top_k, epoch)

    test_top, test_top_k = test(net, test_set, 3)
    print(f"Top move accuracy on test set: {test_top}")
    print(f"Top 3 move average on test set: {test_top_k}")
    writer.add_scalar("Test_accuracy/top_move_accuracy", test_top, epoch)
    writer.add_scalar("Test_accuracy/top_k_move_accuracy", test_top_k, epoch)

    epochTime = time.perf_counter() - start
    print(f"\nEpoch {epoch + 1} time: {epochTime:.4f}s\n")

input("Write new weights to file?")
print("New weights have been saved")
torch.save(net.state_dict(), "alpha-mcts/net/temp.pth")

# Write to tensorboard
writer.flush()
writer.close()