# AlphaZero
Implementation of AlphaZero algorithm with games Connect Four and Gomoku (15x15 Tic-Tac-Toe).

The code is from [This tutorial](https://youtu.be/wuSQpLinRB4?si=4fnr8y4EhGMbSKNA) with slight modifications.

AlphaZero is a reinforcement learning algorithm that uses self-play with Monte Carlo Tree Search (MCTS. It generates games by playing against itself, using MCTS to evaluate moves. The resulting games and their outcomes are used to train and improve a neural network, which again is used in MCTS, creating a cycle of improvement.
