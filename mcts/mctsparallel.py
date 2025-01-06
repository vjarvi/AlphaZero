import numpy as np
import math
from mcts.node import Node

class MCTSParallel:
    """
    Performs Monte Carlo Tree Search (MCTS) in parallel.

    Attributes:
    - game: An instance of the game class defining game mechanics and rules.
    - args: A dictionary of arguments/hyperparameters for training and MCTS.
    - model: The neural network model used to predict policy and value outputs.
    """
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        """
        states: np.ndarray shape of (parallelgames, rows, columns)
        spGames: list of SPG objects

        Performs MCTS simulations in parallel.
        """

        # get predictions (policy) and add noise
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

        # normalize and mask invalid moves.
        game_name = repr(game)
        if game_name == "Gomoku":
          valid_moves = (states == 0).reshape(states.shape[0], -1).astype(np.uint8)
        elif game_name == "ConnectFour":
          valid_moves = (states[:, 0, :] == 0).astype(np.uint8)

        policy *= valid_moves
        policy /= np.sum(policy, axis=1, keepdims=True)

        # initialize root nodes
        for i, spg in enumerate(spGames):
            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(policy[i])

        for search in range(self.args['num_mstc_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)

                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

            if expandable_spGames:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()

                if game_name == "Gomoku":
                    valid_moves = (states == 0).reshape(states.shape[0], -1).astype(np.uint8)
                elif game_name == "ConnectFour":
                    valid_moves = (states[:, 0, :] == 0).astype(np.uint8)

                policy *= valid_moves
                policy /= np.sum(policy, axis=1, keepdims=True)

                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                node.expand(policy[i])
                node.backpropagate(value[i])
