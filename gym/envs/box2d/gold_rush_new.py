import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pygame

import gym
from gym import spaces

# 目前主要地图有两个，在单场比赛中，障碍物的位置不变，炸弹和金币的数量和位置会改变。
# 其中 0表示非障碍物，-1表示障碍物。
mazes = {
    "maze1": np.stack([
        # Channel 0: Agent position (初始为空，会在 reset 中设置)
        np.zeros((17, 17), dtype=np.float32),
        
        # Channel 1: Coins (初始为空)
        np.zeros((17, 17), dtype=np.float32),
        
        # Channel 2: Obstacles (原迷宫障碍物)
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32),
        
        # Channel 3: Bombs (初始为空)
        np.zeros((17, 17), dtype=np.float32),
        
        # Channel 4: Last move (初始为空)
        np.zeros((17, 17), dtype=np.float32)
    ], axis=0),

    "maze2": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, -1, -1, 0, -1, -1, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -0, 0, 0],
        [0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, -1, -1, 0, -1, -1, 0, 0, -1, -0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0, -0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
}

# 每 20 轮，npc 会在固定位置掉落一定数量的金币，金币的数量可能是3、6等，暂时使用固定值3，后续会更新。
npc_coin_positions = {
    "maze1": [
        (0, 5),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 10),
        (1, 11),
        (0, 11),
        (5, 0),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (10, 1),
        (11, 1),
        (11, 0),
        (5, 16),
        (5, 15),
        (6, 15),
        (7, 15),
        (8, 15),
        (9, 15),
        (10, 15),
        (11, 15),
        (11, 16),
        (15, 5),
        (15, 6),
        (15, 7),
        (15, 8),
        (15, 9),
        (15, 10),
        (15, 11),
        (16, 11),
        (16, 5),
    ],
    "maze2": [
        (0, 8),
        (1, 8),
        (2, 8),
        (2, 9),
        (2, 10),
        (2, 11),
        (2, 12),
        (2, 13),
        (1, 13),
        (1, 14),
        (1, 15),
        (2, 15),
        (3, 15),
        (4, 15),
        (5, 15),
        (6, 15),
        (7, 15),
        (8, 15),
        (8, 16),
        (8, 0),
        (8, 1),
        (9, 1),
        (10, 1),
        (11, 1),
        (12, 1),
        (13, 1),
        (14, 1),
        (15, 1),
        (15, 2),
        (15, 3),
        (14, 3),
        (14, 4),
        (14, 5),
        (14, 6),
        (14, 7),
        (14, 8),
        (15, 8),
        (16, 8),
    ],
}

# 可视化部分的代码。
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50
GRID_CELL_SIZE = 40
GRID_BG = (150, 220, 255)  # Light blue
COIN_COLOR = (255, 215, 0)  # Gold


class GoldRushNew(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        maze: str = "maze1",
        stack_frame: int = 4,
        max_rounds: int = 900,
        render_mode: str = "human",
        has_player2: bool = False,
    ):
        # 游戏会进行的最大回合数，因为这个环境每轮执行一个动作，因此应该设置成 900 个回合
        self.max_rounds = max_rounds + 1
        self.stack_frame = stack_frame
        self.maze_type = maze
        self.maze = np.array(copy.deepcopy(mazes[maze]))
        self.npc_coin_pos = npc_coin_positions[maze]

        # 动作空间可看作一个 tuple(int, int)，第一个值是 player1 的动作，第二个值是 player2 的动作，
        # 0: 上，1: 下，2: 左，3: 右，4: 不动。
        # self.action_space = spaces.MultiDiscrete([5, 5])
        self.action_space = spaces.Discrete(5)

        # 判断玩家2是否存在
        self.has_player2 = has_player2
        self.is_multi_frame = False if stack_frame == 1 else True

        # 状态空间是 grid,两个玩家的位置和金币数。
        # 在使用的时候，需要根据是哪个玩家先将该玩家对应的位置由 -2 改成 -9，然后再传入 MoveDecision() 决策。
        # self.observation_space = spaces.Tuple(
        #     (
        #         spaces.Box(low=0, high=500, shape=(17, 17), dtype=np.float32)
        #         for _ in range(6 * self.stack_frame)
        #     )
        # )
        if self.has_player2:
            self.observation_space = spaces.Box(
                low=0, 
                high=255, 
                shape=(6 * self.stack_frame, 17, 17), 
                dtype=np.float32
            )   
        else:
            self.observation_space = spaces.Box(
                low=0, 
                high=255, 
                shape=(5 * self.stack_frame, 17, 17), 
                dtype=np.float32
            )   

        self.historical_states = []
        self.last_move = 4
        self.num_bombs = 10

        # 两个玩家各自的位置， [player1, player2]
        self.agent_positions: List[Tuple[int, int]] = [(0, 0), (16, 16)]
        self.last_positions: List[Tuple[int, int]] = [(0, 0), (16, 16)]

        self.golds: list[int] = [0, 0]  # 表示两个各自玩家的金币数 [player1, player2]

        # 单读保存金币和炸弹的位置方便可视化
        # 表示地图中所有的金币位置和数值
        self.coins: Dict[Tuple[int, int], int] = dict()

        # 表示地图中所有炸弹的位置
        self.bombs: Set[Tuple[int, int]] = set()

        # 吃到炸弹后会损失的金币的比例，目前观测是 0.1，但是具体的值需要到初赛时才会公布。
        self.penalty = 0.1

        self.round = 1  # 当前是第几回合
        self.turn = 0  # 当前是哪个玩家，0: player1, 1: player2

        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        重置环境
        """
        self.golds = [0, 0]
        self.coins.clear()
        self.bombs.clear()
        self.round = 1
        self.turn = 0
        self.last_move = 4
        self.agent_positions = [(0, 0), (16, 16)]
        self.last_positions = [(0, 0), (16, 16)]
        self.maze = np.array(copy.deepcopy(mazes[self.maze_type]))
        # self.maze[0][0] = -2
        # self.maze[16][16] = -2
        self.maze[0, 0, 0] = 1  # palyer1
        if self.has_player2:
            self.maze[1, 16, 16] = 1  # palyer2
        self._flush_npc()
        self._flush_coins()
        self._flush_bomb()

        # Initialize historical states.
        obs = self._get_obs_frame()
        self.historical_states = [obs for _ in range(self.stack_frame)]

        if self.is_multi_frame:
            return self._get_obs(), {}
        else:
            return self._get_obs_frame(), {}


    def step(self, action):
        """
        这个函数只会执行一个玩家的动作，action 的第一个变量表示是哪个玩家,0: player1, 1:player2，
        第二个变量表示具体的动作，范围是 [0, 4]，具体含义和游戏规则一样
        """
        agent_id = 0
        move = action

        # 更新地图
        # if self.turn == 0:
        #     if self.round % 20 == 1:
        #         # 每20个回合会有 npc 随机掉落金币
        #         self._flush_npc()
        #     self.round += 1

        #     # 在每个回合开始的时候，会先掉落金币，然后在把地图传给玩家
        #     self._flush_coins()
        # self.turn = 1 if self.turn == 0 else 0

        if self.round % 60 == 1 and self.round != 1:
            # 每20个回合会有 npc 随机掉落金币
            self._flush_npc()
            self._flush_bomb()
        self.round += 1
        if self.round % 3 == 1 and self.round != 1:
            self._flush_coins()

        # rewards 目前就是吃到或者损失的金币数
        rewards = 0

        done = False
        info = dict()  # gym 框架需要，暂时没用

        # 对应 [0, 4] 在地图中的变化，第一个元素是行的变化，第二个是列的变化
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

        # 记录上一步的旧信息
        if self.has_player2:
            r, c = self.last_positions[agent_id]
            self.maze[5, r, c] = 0
        else:
            r, c = self.last_positions[agent_id]
            self.maze[4, r, c] = 0

        r, c = self.agent_positions[agent_id]
        new_r = np.clip(r + moves[move][0], 0, 16)
        new_c = np.clip(c + moves[move][1], 0, 16)

        # 只有新的位置是非障碍物和玩家才有用
        self.last_positions = self.agent_positions.copy()
        old_r, old_c = self.last_positions[agent_id]
        if self._is_position_valid(new_r, new_c):
            # 清除旧位置
            self.maze[agent_id, r, c] = 0
            # 设置新位置
            self.maze[agent_id, new_r, new_c] = 1
            self.agent_positions[agent_id] = (new_r, new_c)
            # 如果新的位置是金币或者炸弹，该进行相应的处理
            if self.has_player2:
                if self.maze[2, new_r, new_c] > 0:  # 金币
                    rewards = self.maze[2, new_r, new_c]
                    self.maze[2, new_r, new_c] = 0
                    self.coins.pop((new_r, new_c))
                elif self.maze[4, new_r, new_c] == 1:  # 炸弹
                    rewards = -int(self.golds[agent_id] * self.penalty)
                    self.maze[4, new_r, new_c] = 0
                    self.bombs.remove((new_r, new_c))
            else:
                if self.maze[1, new_r, new_c] > 0:  # 金币
                    rewards = self.maze[1, new_r, new_c]
                    self.maze[1, new_r, new_c] = 0
                    self.coins.pop((new_r, new_c))
                elif self.maze[3, new_r, new_c] == 1:  # 炸弹
                    rewards = -int(self.golds[agent_id] * self.penalty)
                    self.maze[3, new_r, new_c] = 0
                    self.bombs.remove((new_r, new_c))

            self.last_move = move
        else:
            rewards = -3
            self.last_move = 4

        self.golds[agent_id] += rewards

        # 更新新的上一步信息
        if self.has_player2:
            r, c = self.last_positions[agent_id]
            self.maze[5, old_r, old_c] = self.last_move + 1
        else:
            r, c = self.last_positions[agent_id]
            self.maze[4, old_r, old_c] = self.last_move + 1

        if self.round == self.max_rounds:
            done = True
            # score, opponent_score = self.golds[agent_id], self.golds[1 - agent_id]
            # rewards = 10000 if score > opponent_score else -10000

        self.historical_states.append(self._get_obs_frame())
        self.historical_states.pop(0)

        if self.is_multi_frame:
            return self._get_obs(), rewards, done, False, info
        else:
            a = self._get_obs_frame()
            return self._get_obs_frame(), rewards, done, False, info



    def _get_obs_frame(self) -> np.ndarray:
        if self.has_player2:
            obs = np.stack([
                self.maze[0],  # agent
                self.maze[1],  # opponent
                self.maze[2],  # coins
                self.maze[3],  # obstacles
                self.maze[4],  # bombs
                self.maze[5],  
            ], axis=0)
            return obs
        else:
            obs = np.stack([
                self.maze[0],  # agent
                self.maze[1] / 25,  # coins
                self.maze[2],  # obstacles
                self.maze[3],  # bombs
                self.maze[4] / 5,
            ], axis=0)
            return obs

    def _get_obs(self) -> np.ndarray:
        return np.vstack(self.historical_states)
    
    def _is_position_valid(self, r: int, c: int) -> bool:
        # 判断新的位置是否是非障碍物和玩家
        if self.has_player2:
            valid = (
                self.maze[3, r, c] != 1
                and self.maze[0, r, c] != 1 
                and self.maze[1, r, c] != 1
            )
        
        else:
            valid = (
                self.maze[2, r, c] != 1
                and self.maze[0, r, c] != 1 
            )
        return valid

# 以下方法在有player2时需要更改
    def _flush_coins(self):
        """
        在每回合开始前更新地图中的金币
        """
        cnt = 0
        while cnt < 1:
            r, c = random.randint(4, 12), random.randint(4, 12)
            if self._is_position_valid(r, c):
                value = random.randint(3, 25)
                self.maze[1, r, c] = value
                self.coins.update({(r, c): value})
                cnt += 1

    def _flush_npc(self):
        for r, c in self.npc_coin_pos:
            if self._is_position_valid(r, c):
                self.maze[1, r, c] = 3
                self.coins.update({(r, c): 3})

    def _flush_bomb(self):
        self.bombs.clear()
        self.maze[3] = 0
        cnt = 0
        while cnt < self.num_bombs:
            r, c = random.randint(0, 16), random.randint(0, 16)
            if self._is_position_valid(r, c) and self.maze[1, r, c] == 0:
                self.bombs.add((r, c))
                self.maze[3, r, c] = 1
                cnt += 1
