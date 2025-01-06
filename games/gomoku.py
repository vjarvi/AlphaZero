import numpy as np

class Gomoku():
    """
    A class representing the gomoku game, played on a 15x15 board.
    The objective is to place five of your marks in a horizontal, vertical,
    or diagonal row. Default board size is 15x15.
    """

    def __init__(self, rows=15, columns=15):
      self.action_size = rows * columns
      self.columns = columns
      self.rows = rows
      self.in_a_row = 5 # number of consecutive marks required to win

    def __repr__(self):
      return "Gomoku"

    def get_initial_state(self):
      return np.zeros((self.rows, self.columns))

    def get_next_state(self, state, action, player):
      row = action // self.columns
      column = action % self.columns
      state[row, column] = player
      return state

    def get_valid_moves(self, state):
      return ((state == 0).astype(np.uint8))

    def check_win(self, state, action):
      if action == None:
        return False

      row = action // self.columns
      column = action % self.columns
      player = state[row, column]

      def count(offset_row, offset_column):
        for i in range(1, self.in_a_row):
          r = row + offset_row * i
          c = column + offset_column * i
          if (
            r < 0
            or r >= self.rows
            or c < 0
            or c >= self.columns
            or state[r][c] != player
            ):
              return i - 1
        return self.in_a_row - 1

      return (
        (count(1, 0) + count(-1, 0)) >= self.in_a_row - 1 # vertical
        or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
        or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
        or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
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
