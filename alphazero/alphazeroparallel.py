import pickle
import random
import torch
import numpy as np
from mcts.mctsparallel import MCTSParallel

class AlphaZeroParallel:
    """
    Runs AlphaZero algorithm in parallel.

    Attributes:
    - model: The neural network model used to predict policy and value outputs.
    - optimizer: The optimizer used for updating model weights.
    - policy_loss_fn: Loss function for the policy head of the model.
    - value_loss_fn: Loss function for the value head of the model.
    - game: An instance of the game class defining game mechanics and rules.
    - args: A dictionary of arguments/hyperparameters for training and MCTS.
    - verbose: A boolean for showning model accuracy.
    - mcts: An instance of the MCTSParallel class used for simulating moves.
    """

    def __init__(self, model, optimizer, policy_loss_fn, value_loss_fn, game, args, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.policy_loss_fn = policy_loss_fn
        self.value_loss_fn = value_loss_fn
        self.game = game
        self.args = args
        self.verbose = verbose
        self.mcts = MCTSParallel(game, args, model)

    def get_canonical_boards(self,state, action_probs, player):
        """
        state: np.ndarray shape of (rows, columns)
        action_probs: np.ndarray shape of (action_size)
        player: int

        returns:
        list of tuples (state, action_probs, player)

        Does data augmentation by flipping the state and action_probs.
        """
        game_name = repr(game)

        # gomoku allows for more configurations of the board than connect four.
        if game_name == "Gomoku":
            memory = []
            current_state = state.astype(np.int8)
            current_probs = action_probs.astype(np.float32)

            memory.extend([
                [current_state, current_probs, player],
                [np.flip(current_state, axis=0),
                np.flip(current_probs.reshape(15, 15), axis=0).reshape(225),
                player]
            ])

            for i in range(3):
                current_state = np.rot90(current_state)
                current_probs = current_probs.reshape(15, 15)
                current_probs = np.rot90(current_probs)
                current_probs = current_probs.reshape(225)

                # Create flipped versions
                memory.extend([
                    [current_state, current_probs, player],
                    [np.flip(current_state, axis=0),
                    np.flip(current_probs.reshape(15, 15), axis=0).reshape(225),
                    player]
                ])

        elif game_name == "ConnectFour":
            memory = [
              [state.astype(np.int8), action_probs.astype(np.float32), player],
              [np.flip(state.astype(np.int8), axis=1),
              np.flip(action_probs.astype(np.float32), axis=0),
              player]
            ]

        return memory

    def selfPlay(self):
        """
        returns:
        list of tuples (state, action_probs, player)

        Generates training data by simulating games using Monte Carlo Tree Search (MCTS).
        """

        return_memory = []
        player = 1
        spGames = (SPG(self.game) for _ in range(self.args['num_parallel_games']))
        active_games = list(spGames)

        while active_games:

            #monitor progress
            print(f"{len(active_games)} parallel games left")

            states = np.stack([spg.state for spg in active_games])
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, active_games)

            for i in range(len(active_games))[::-1]:
                spg = active_games[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.extend((self.get_canonical_boards(spg.root.state, action_probs, player)))
                #spg.memory.append((spg.root.state, action_probs, player)) (adding to memory without get_canonical_boards)

                temperature_action_probs = (action_probs ** (1 / self.args['temperature']))
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)
                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del active_games[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        """
        memory: list of tuples (state, action_probs, value)

        training loop for neural network
        """

        policy_losses = []
        value_losses = []
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx+self.args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = self.policy_loss_fn(out_policy, policy_targets)
            value_loss = self.value_loss_fn(out_value, value_targets)
            loss = args['policy_value_bias'] * policy_loss + (1 - args['policy_value_bias']) * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

            # Visualize loss
            if self.verbose and batchIdx % 50 == 0:
                print(f"batch {batchIdx // self.args['batch_size']}: policy_loss mean: {np.mean(policy_losses):.3f} | value loss mean: {np.mean(value_losses):.3f}")

    def learn(self):
        """
        Iterates between selfplay and model training
        """

        for iteration in range(self.args['num_iterations']):
            memory = []

            # generate games
            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory.extend(self.selfPlay())

            if memory != []:
              with open(f"iteration_{iteration}_memory.plk", "wb") as file:
                pickle.dump(memory, file)

            # train the model
            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")

            torch.cuda.empty_cache() if torch.cuda.is_available() else None


class SPG:
    """
    'SelfPlayGame' class for managing game states and trees during selfplay.
    """

    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

  
