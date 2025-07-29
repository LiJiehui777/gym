import pdb
import time

import numpy as np
from loguru import logger

import gym
from randomBot import Player

if __name__ == "__main__":
    env = gym.make("GoldRush-v0", maze="maze2", stack_frame=6)
    obs, _ = env.reset()
    env.render()

    player1, player2 = Player(), Player()
    done = False
    round = 1

    while not done:
        grid, golds = env.display_info(agent_id=0)
        moves1 = player1.MoveDecision(grid.tolist(), golds[0], golds[1])

        grid, golds = env.display_info(agent_id=1)
        moves2 = player2.MoveDecision(grid.tolist(), golds[1], golds[0])

        for i in range(3):
            logger.info(f"Player 1: round {round} | move = {moves1[i]}")
            obs, rewards, done, _, info = env.step((0, moves1[i]))
            print(f"obs.shape = {obs.shape}")
            env.render()
            time.sleep(0.1)

            logger.info(f"Player 2: round {round} | move = {moves2[i]}")
            obs, rewards, done, _, info = env.step((1, moves2[i]))
            print(f"obs.shape = {obs.shape}")

            if done:
                break
            env.render()
            time.sleep(0.1)

            if done:
                break
            round += 1
