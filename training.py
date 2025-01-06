import torch
import torch.nn as nn
from model.resnet import ResNet
from games.gomoku import Gomoku
from games.tictactoe import TicTacToe
from games.connectfour import ConnectFour
from alphazeroparallel.alphazeroparallel import AlphaZeroParallel

game = ConnectFour()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

model = ResNet(game, num_resBlocks=9, num_hidden=128, device=device)
optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# loading a model and optimizer
#model.load_state_dict(torch.load("model_3_ConnectFour.pt", map_location=device))
#optim.load_state_dict(torch.load("optimizer_3_ConnectFour.pt", map_location=device))

value_loss_fn = nn.MSELoss()
policy_loss_fn = nn.CrossEntropyLoss()

# hyperparameters
args = {
    'C': 2,                          # UCB exploration constant
    'num_mstc_searches': 600,        # Number of MCTS simulations per move
    'num_iterations': 8,             # Training iterations
    'num_selfPlay_iterations': 250,  # Self-play games per iteration
    'num_parallel_games': 250,       # Number of games to play in parallel
    'num_epochs': 4,                 # Training epochs per iteration
    'batch_size': 128,              # Training batch size
    'temperature': 1.25,                # Temperature for action selection
    'policy_value_bias': 0.5,        # Balance between policy and value loss
    'dirichlet_epsilon': 0.3,       # Exploration noise weight
    'dirichlet_alpha': 0.2         # Dirichlet distribution parameter
}

alphaZero = AlphaZeroParallel(model, optim, policy_loss_fn, value_loss_fn, game, args)
alphaZero.learn()

