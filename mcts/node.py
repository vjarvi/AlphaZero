import numpy as np
import math

class Node:
    """
    Represents a node in a Monte Carlo Tree Search (MCTS) used for decision-making
    in AlphaZero. Each node corresponds to a specific state of the game and contains information about its children, parent, visit counts and

    Attributes:
    - game: The game object, providing methods to compute game states and transitions.
    - args: A dictionary of arguments/hyperparameters
    - state: The current state of the game associated with this node.
    - parent: The parent node of this node. None if it is the root.
    - action_taken: The action that led to this node's state from the parent node.
    - prior: The prior probability of selecting this node.
    - children: A list of child nodes representing possible next states.
    - visit_count: The number of times this node has been visited during MCTS simulations.
    - value_sum: The cumulative value from simulations passing through this node.
    """

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            value = self.game.get_opponent_value(value)
            self.parent.backpropagate(value)
