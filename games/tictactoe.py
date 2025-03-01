import numpy as np

class TicTacToe():
    def __init__(self, rows=3, columns=3):
      self.action_size = rows * columns
      self.columns = columns
      self.rows = rows

    def __repr__(self):
      return "TicTacToe"

    def get_initial_state(self):
      return np.zeros((self.rows, self.columns))

    def get_next_state(self, state, action, player):
      row = action // self.columns
      column = action % self.columns
      state[row, column] = player
      return state

    def get_valid_moves(self, state):
      return ((state == 0).astype(np.uint8))

    def canonical_boards(self):
      pass

    def check_win(self, state, action):
      if action == None:
        return False

      row = action // self.columns
      column = action % self.columns
      player = state[row, column]

      return (
          np.sum(state[row, :]) == player * self.columns
          or np.sum(state[:, column]) == player * self.rows
          or np.sum(np.diag(state)) == player * self.rows
          or np.sum(np.diag(np.flip(state, axis=0))) == player * self.rows
      )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
      return -value

    def change_perspective(self, state, player):
      return state * player

    def get_encoded_state(self, state):
      encoded_state = np.stack(
          (state == -1, state == 0, state == 1)
      ).astype(np.float32)

      if len(state.shape) == 3:
        encoded_state = np.swapaxes(encoded_state, 0, 1)

      return encoded_state
