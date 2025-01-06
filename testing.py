import torch
import torch.nn as nn
from model.resnet import ResNet
from games.gomoku import Gomoku
from games.tictactoe import TicTacToe
from games.connectfour import ConnectFour
from mcts.mcts import MCTS

game = ConnectFour()
player = 1

args = {
    'C': 2,
    'num_mcts_searches': 500,
    'dirichlet_epsilon': 0.0,
    'dirichlet_alpha': 0.1
}

model = ResNet(game, 9, 128, device)
#model1.load_state_dict(torch.load("model_3_ConnectFour (1).pt", map_location=device))
model.eval()

mcts = MCTS(game, args, model)


state = game.get_initial_state()


while True:
    print(state)

    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue

    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs, pos_value, value = mcts2.search(neutral_state)
        action = np.argmax(mcts_probs)

    state = game.get_next_state(state, action, player)

    value, is_terminal = game.get_value_and_terminated(state, action)

    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break

    player = game.get_opponent(player)


