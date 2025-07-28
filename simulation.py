import pdb
import time

import numpy as np
from loguru import logger

import gym
from randomBot import Player

if __name__ == "__main__":
    env = gym.make("GoldRush-v0", maze="maze1")
    (grid, positions, golds), _ = env.reset()
    env.render()

    player1, player2 = Player(), Player()

    while True:
        grid = grid.reshape(17, 17)
        r, c = positions[0]
        grid[r][c] = -9
        moves1 = player1.MoveDecision(grid.tolist(), golds[0], golds[1])
        grid[r][c] = -2

        r, c = positions[1]
        grid[r][c] = -9
        moves2 = player2.MoveDecision(grid.tolist(), golds[1], golds[0])
        grid[r][c] = -2

        for i in range(3):
            logger.info(f"Player 1: {moves1[i]}")
            (grid, positions, golds), rewards, done, _, info = env.step((0, moves1[i]))
            env.render()
            time.sleep(0.1)

            logger.info(f"Player 1: {moves1[i]}")
            (grid, positions, golds), rewards, done, _, info = env.step((1, moves2[i]))
            env.render()
            time.sleep(0.1)
