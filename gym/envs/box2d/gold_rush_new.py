import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pygame

import gym
from gym import spaces
import math

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
        
        # Channel 4: Visited_map (初始为空)
        np.zeros((17, 17), dtype=np.float32)
    ], axis=0),

    "maze2": np.stack([
        # Channel 0: Agent position (初始为空，会在 reset 中设置)
        np.zeros((17, 17), dtype=np.float32),
        
        # Channel 1: Coins (初始为空)
        np.zeros((17, 17), dtype=np.float32),
        
        # Channel 2: Obstacles (原迷宫障碍物)
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32),
        
        # Channel 3: Bombs (初始为空)
        np.zeros((17, 17), dtype=np.float32),
        
        # Channel 4: Visited_map (初始为空)
        np.zeros((17, 17), dtype=np.float32)
    ], axis=0),
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
        stack_frame: int = 3,
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
        self.steps_to_eat_coin = 0

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
        self.num_bombs = 20

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
        maze_choice = random.choice(["maze1", "maze2"])
        self.maze = np.array(copy.deepcopy(mazes[self.maze_type]))
        # self.maze[0][0] = -2
        # self.maze[16][16] = -2
        self.maze[0, 0, 0] = 1  # palyer1
        self.maze[4, 0, 0] = 1
        if self.has_player2:
            self.maze[1, 16, 16] = 1  # palyer2
        self.npc_coin_pos = npc_coin_positions[maze_choice]
        self._flush_npc()
        self._flush_coins()
        self._flush_bomb()
        self.steps_to_eat_coin = 0

        # Initialize historical states.
        obs = self._get_obs_frame()
        self.historical_states = [obs for _ in range(self.stack_frame)]
        # obs_channel_agent = self._get_obs_frame_channel_agent()
        # self.historical_agent_states = [obs_channel_agent for _ in range(self.stack_frame)]

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
        self.steps_to_eat_coin += 1

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

            # 更新player的位置信息
            self._flush_agent_location()

            obs = self._get_obs_frame()
            self.historical_states = [obs for _ in range(self.stack_frame)]
        self.round += 1
        if self.round % 9 == 1 and self.round != 1:
            self._flush_coins()

        # rewards 目前就是吃到或者损失的金币数
        rewards = 0
        reward_coin = 0
        reward_bomb = 0
        reward_obstacle = 0
        reward_step = -1  # 走一步就需要-1的惩罚

        done = False
        info = dict()  # gym 框架需要，暂时没用

        # 对应 [0, 4] 在地图中的变化，第一个元素是行的变化，第二个是列的变化
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

        r, c = self.agent_positions[agent_id]
        new_r = np.clip(r + moves[move][0], 0, 16)
        new_c = np.clip(c + moves[move][1], 0, 16)

        # 接近价值/步数最大的金币
        old_target = self._distance_to_nearest_coin(r, c)
        new_target = self._distance_to_nearest_coin(new_r, new_c)
        if old_target and new_target and old_target[0] == new_target[0]:
            rewards += (old_target[1] - new_target[1])

        # 只有新的位置是非障碍物和玩家才有用
        self.last_positions = self.agent_positions.copy()
        old_r, old_c = self.last_positions[agent_id]
        if self._is_position_valid(new_r, new_c):
            # 清除旧位置
            self.maze[agent_id, r, c] = 0
            # 设置新位置
            self.maze[agent_id, new_r, new_c] = 1
            self.agent_positions[agent_id] = (new_r, new_c)

            # 更新player走过的位置信息
            self.maze[4, new_r, new_c] = 1

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
                    reward_coin = self.maze[1, new_r, new_c] / self.steps_to_eat_coin
                    # rewards = self.maze[1, new_r, new_c] / math.sqrt(self.steps_to_eat_coin)
                    self.steps_to_eat_coin = 0
                    self.golds[agent_id] += self.maze[1, new_r, new_c]
                    self.maze[1, new_r, new_c] = 0
                    self.coins.pop((new_r, new_c))
                elif self.maze[3, new_r, new_c] == 1:  # 炸弹
                    reward_bomb = -20
                    r = -int(self.golds[agent_id] * self.penalty)
                    # rewards = -20
                    self.maze[3, new_r, new_c] = 0
                    self.golds[agent_id] += r
                    self.bombs.remove((new_r, new_c))
                # else:
                #     rewards = -1

        else:
            reward_obstacle = -10
            # rewards = -5

        rewards = reward_bomb + reward_coin + reward_obstacle + reward_step
        # nearest_coin_dist = self._distance_to_nearest_coin(agent_id)
        # if nearest_coin_dist < 3:
        #     rewards += (1 - nearest_coin_dist / 32)
        # else:
        #     rewards += 0.2 * (1 - nearest_coin_dist / 32)
        # rewards += 0.5 * (1 - nearest_coin_dist)

        if self.round == self.max_rounds:
            done = True
            print("coins: ", self.golds[agent_id])
            # score, opponent_score = self.golds[agent_id], self.golds[1 - agent_id]
            # rewards = 10000 if score > opponent_score else -10000

        self.historical_states.append(self._get_obs_frame())
        self.historical_states.pop(0)
        # self.historical_agent_states.append(self._get_obs_frame_channel_agent())
        # self.historical_agent_states.pop(0)

        if self.is_multi_frame:
            return self._get_obs(), rewards, done, False, info
        else:
            return self._get_obs_frame(), rewards, done, False, info
        
    def display_info(self, agent_id: int):
        maze_new = np.zeros((17, 17), dtype=np.float32)
        for j in range(17):
            for k in range(17):
                if self.maze[1, j, k] >= 1:
                    maze_new[j, k] = self.maze[1, j, k]
                elif self.maze[2, j, k] == 1:
                    maze_new[j, k] = -1
                elif self.maze[3, j, k] == 1:
                    maze_new[j, k] = -3

        r, c = self.agent_positions[agent_id]
        maze_new[r][c] = -9

        return maze_new, copy.deepcopy(self.golds)

    def _distance_to_nearest_coin(self, r, c):
        r, c = r, c
        ratio = float('-inf')
        result_dist = 0
        
        for coin_pos in self.coins:
            dist = abs(r - coin_pos[0]) + abs(c - coin_pos[1])  # 曼哈顿距离
            coin = self.maze[1, coin_pos[0], coin_pos[1]]
            if ratio < coin / dist:
                ratio = coin / dist
                result_dist = dist
                location = (coin_pos[0], coin_pos[1])
        
        return [location, result_dist]

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
                self.maze[0],
                self.maze[1],  # coins
                self.maze[2],  # obstacles
                self.maze[3],  # bombs
                self.maze[4],  # visited_map
            ], axis=0)
            obs_min = obs.min() 
            obs_max = obs.max() 
            return (obs - obs_min) / (obs_max - obs_min + 1e-8)

    def _get_obs(self) -> np.ndarray:
        return np.vstack(self.historical_states)

    def global_minmax_normalize(obs):
        """将整个obs张量归一化到[0,1]范围"""
        obs_min = obs.min() 
        obs_max = obs.max() 
        return (obs - obs_min) / (obs_max - obs_min + 1e-8)
    
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

    def _get_coins(self):
        return self.golds[0]

# 以下方法在有player2时需要更改
    def _flush_coins(self):
        """
        在每回合开始前更新地图中的金币
        """
        cnt = 0
        while cnt < 1:
            r, c = random.randint(4, 12), random.randint(4, 12)
            if self._is_position_valid(r, c) and self.maze[3, r, c] == 0:
                value = random.randint(3, 25)
                self.maze[1, r, c] = value
                self.coins.update({(r, c): value})
                cnt += 1

    def _flush_npc(self):
        self.maze[1, :4, :] = 0
        self.maze[1, 13:, :] = 0
        self.maze[1, :, :4] = 0
        self.maze[1, :, 13:] = 0
        maze_choice = random.choice(["maze1", "maze2"])
        self.npc_coin_pos = npc_coin_positions[maze_choice]
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

    def _flush_agent_location(self):
        # 重置player走过的位置信息 
        self.maze[4] = 0
        r, c = self.agent_positions[0] 
        self.maze[4, r, c] = 1

    def render(self):
        """
        渲染地图和游戏状态（支持单/双玩家模式）
        """
        # 初始化pygame显示
        if not hasattr(self, 'screen'):
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((900, 900))
            
            # 加载图像资源（失败时用彩色方块替代）
            try:
                self.box_image = pygame.image.load("box.png").convert_alpha()
                self.bomb_image = pygame.image.load("bomb.png").convert_alpha()
                self.player1_image = pygame.image.load("player1.png").convert_alpha()
                self.player2_image = pygame.image.load("player2.png").convert_alpha()
            except:
                # 创建替代图像
                self.box_image = pygame.Surface((GRID_CELL_SIZE, GRID_CELL_SIZE))
                self.box_image.fill((139, 69, 19))  # 棕色障碍物
                self.bomb_image = pygame.Surface((GRID_CELL_SIZE, GRID_CELL_SIZE))
                self.bomb_image.fill((255, 0, 0))    # 红色炸弹
                self.player1_image = pygame.Surface((GRID_CELL_SIZE, GRID_CELL_SIZE))
                self.player1_image.fill((0, 0, 255))  # 蓝色玩家1
                self.player2_image = pygame.Surface((GRID_CELL_SIZE, GRID_CELL_SIZE))
                self.player2_image.fill((255, 0, 255)) # 粉色玩家2

        # 清空画布
        self.screen.fill((0, 0, 0))

        # 绘制网格和对象
        for r in range(17):
            for c in range(17):
                rect = pygame.Rect(
                    GRID_OFFSET_X + c * GRID_CELL_SIZE,
                    GRID_OFFSET_Y + r * GRID_CELL_SIZE,
                    GRID_CELL_SIZE,
                    GRID_CELL_SIZE
                )
                
                # 绘制背景格子
                pygame.draw.rect(self.screen, GRID_BG, rect)
                
                # 绘制障碍物（Channel 2）
                if self.maze[2, r, c] == 1:
                    self.screen.blit(self.box_image, rect)
                
                # 绘制炸弹（Channel 3）
                if (r, c) in self.bombs:
                    self.screen.blit(self.bomb_image, rect)
                
                # 绘制金币（Channel 1）
                if (r, c) in self.coins:
                    pygame.draw.circle(
                        self.screen, COIN_COLOR, rect.center, 
                        GRID_CELL_SIZE // 2 - 4
                    )
                    # 显示金币数值
                    font = pygame.font.SysFont(None, 24)
                    text = font.render(
                        str(self.coins[(r, c)]), True, (0, 0, 0)
                    )
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
                
                # 绘制网格线
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        # 绘制玩家1
        p1_r, p1_c = self.agent_positions[0]
        p1_rect = pygame.Rect(
            GRID_OFFSET_X + p1_c * GRID_CELL_SIZE,
            GRID_OFFSET_Y + p1_r * GRID_CELL_SIZE,
            GRID_CELL_SIZE,
            GRID_CELL_SIZE
        )
        self.screen.blit(self.player1_image, p1_rect)

        # 绘制玩家2（如果存在）
        if self.has_player2:
            p2_r, p2_c = self.agent_positions[1]
            p2_rect = pygame.Rect(
                GRID_OFFSET_X + p2_c * GRID_CELL_SIZE,
                GRID_OFFSET_Y + p2_r * GRID_CELL_SIZE,
                GRID_CELL_SIZE,
                GRID_CELL_SIZE
            )
            self.screen.blit(self.player2_image, p2_rect)

        # 显示游戏状态信息
        font = pygame.font.SysFont(None, 36)
        info_text = [
            f"Round: {self.round}/{self.max_rounds}",
            f"Player1 Gold: {self.golds[0]}",
            f"Player2 Gold: {self.golds[1]}" if self.has_player2 else ""
        ]
        
        for i, text in enumerate(filter(None, info_text)):
            text_surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (20, 20 + i * 40))

        pygame.display.flip()
