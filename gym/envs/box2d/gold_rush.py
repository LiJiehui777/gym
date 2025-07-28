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
    "maze1": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
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


class GoldRush(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self, maze: str = "maze1", max_rounds: int = 900, render_mode: str = "human"
    ):
        # 游戏会进行的最大回合数，因为这个环境每轮执行一个动作，因此应该设置成 900 个回合
        self.max_rounds = max_rounds + 1
        self.maze_type = maze
        self.maze = copy.deepcopy(mazes[maze])
        self.npc_coin_pos = npc_coin_positions[maze]

        # 动作空间可看作一个 tuple(int, int)，第一个值是 player1 的动作，第二个值是 player2 的动作，
        # 0: 上，1: 下，2: 左，3: 右，4: 不动。
        self.action_space = spaces.MultiDiscrete([5, 5])

        # 状态空间是 grid,两个玩家的位置和金币数。
        # 在使用的时候，需要根据是哪个玩家先将该玩家对应的位置由 -2 改成 -9，然后再传入 MoveDecision() 决策。
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=-9, high=500, shape=(17 * 17,), dtype=np.int32),
                spaces.Tuple(
                    (
                        spaces.Tuple((spaces.Discrete(17), spaces.Discrete(17))),
                        spaces.Tuple((spaces.Discrete(17), spaces.Discrete(17))),
                    )
                ),
                spaces.Tuple((spaces.Discrete(5000), spaces.Discrete(5000))),
            )
        )

        # 两个玩家各自的位置， [player1, player2]
        self.agent_positions: List[Tuple[int, int]] = [(0, 0), (16, 16)]

        self.golds: list[int] = [0, 0]  # 表示两个各自玩家的金币数 [player1, player2]

        # 表示地图中所有的金币位置和数值
        self.coins: Dict[Tuple[int, int], int] = dict()

        # 表示地图中所有炸弹的位置
        self.bombs: Set[Tuple[int, int]] = set()

        # 吃到炸弹后会损失的金币的比例，目前观测是 0.1，但是具体的值需要到初赛时才会公布。
        self.penalty = 0.1

        self.round = 1  # 当前是第几回合
        self.turn = 0  # 当前是哪个玩家，0: player1, 1: player2

        # 可视化相关的代码
        self.screen: Optional[pygame.Surface] = None
        self.render_mode = render_mode  # gym 框架需要的成员变量

        # 导入地图中的图片
        image_folder = Path(__file__).parent.parent.joinpath("toy_text", "img")
        self.box_image = pygame.image.load(image_folder.joinpath("box.png"))
        self.box_image = pygame.transform.smoothscale(
            self.box_image, (GRID_CELL_SIZE - 1, GRID_CELL_SIZE - 1)
        )
        self.bomb_image = pygame.image.load(image_folder.joinpath("bomb.png"))
        self.bomb_image = pygame.transform.smoothscale(
            self.bomb_image, (GRID_CELL_SIZE - 1, GRID_CELL_SIZE - 1)
        )
        self.player1_image = pygame.image.load(image_folder.joinpath("cab_front.png"))
        self.player1_image = pygame.transform.smoothscale(
            self.player1_image, (GRID_CELL_SIZE - 1, GRID_CELL_SIZE - 1)
        )
        self.player2_image = pygame.image.load(image_folder.joinpath("elf_down.png"))
        self.player2_image = pygame.transform.smoothscale(
            self.player2_image, (GRID_CELL_SIZE - 1, GRID_CELL_SIZE - 1)
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        重置环境
        """
        self.agent_positions = [(0, 0), (16, 16)]
        self.golds = [0, 0]
        self.coins = {}
        self.round = 1
        self.turn = 0
        self.maze = copy.deepcopy(mazes[self.maze_type])
        self.maze[0][0] = -2
        self.maze[16][16] = -2

        return self._get_obs(), {}

    def step(self, action: Tuple[int, int]):
        """
        这个函数只会执行一个玩家的动作，action 的第一个变量表示是哪个玩家,0: player1, 1:player2，
        第二个变量表示具体的动作，范围是 [0, 4]，具体含义和游戏规则一样
        """
        agent_id, move = action

        # 更新地图
        if self.turn == 0:
            if self.round % 20 == 1:
                # 每20个回合会有 npc 随机掉落金币
                self._flush_npc()
            self.round += 1

            # 在每个回合开始的时候，会先掉落金币，然后在把地图传给玩家
            self._flush_coins()
        self.turn = 1 if self.turn == 0 else 0

        # rewards 目前就是吃到或者损失的金币数
        rewards = 0

        done = False
        info = dict()  # gym 框架需要，暂时没用

        # 对应 [0, 4] 在地图中的变化，第一个元素是行的变化，第二个是列的变化
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

        r, c = self.agent_positions[agent_id]
        new_r = np.clip(r + moves[move][0], 0, 16)
        new_c = np.clip(c + moves[move][1], 0, 16)

        # 只有新的位置是非障碍物和玩家才有用
        if self.maze[new_r][new_c] not in [-1, -2]:
            self.agent_positions[agent_id] = (new_r, new_c)

            self.maze[r][c] = 0
            self.maze[new_r][new_c] = -2

            pos = (new_r, new_c)

            # 如果新的位置是金币或者炸弹，怎进行相应的处理
            if pos in self.coins:
                rewards = self.coins[pos]
                self.coins.pop(pos)
                self.maze[pos[0]][pos[1]] = 0
            elif pos in self.bombs:
                rewards = -int(self.golds[agent_id] * self.penalty)
                self.bombs.remove(pos)
                self.maze[pos[0]][pos[1]] = 0

        if len(self.coins) == 0 or self.round == self.max_rounds:
            done = True

        return self._get_obs(), rewards, done, False, info

    def _get_obs(self):
        """
        返回地图，玩家的位置和金币数量作为状态空间（地图中没有-9，需要手动设置）
        """
        return (
            np.array(self.maze, dtype=np.int32).flatten(),
            copy.deepcopy(tuple(self.agent_positions)),
            copy.deepcopy(tuple(self.golds)),
        )

    def _flush_coins(self):
        """
        在每回合开始前更新地图中的金币
        """
        cnt = 0
        while cnt < 4:
            r, c = random.randint(4, 12), random.randint(4, 12)
            if (r, c) not in self.bombs and self.maze[r][c] != -1:
                self.coins.update({(r, c): random.randint(3, 25)})
                cnt += 1

    def _flush_npc(self):
        self.coins.update({pos: 3 for pos in self.npc_coin_pos})

    def render(self):
        """
        渲染地图
        """
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((900, 900))

        for r in range(17):
            for c in range(17):
                rect = pygame.Rect(
                    GRID_OFFSET_X + c * GRID_CELL_SIZE,
                    GRID_OFFSET_Y + r * GRID_CELL_SIZE,
                    GRID_CELL_SIZE,
                    GRID_CELL_SIZE,
                )
                pygame.draw.rect(self.screen, GRID_BG, rect)

                pos = (r, c)
                if self.maze[r][c] == -1:
                    self.screen.blit(self.box_image, rect)
                elif pos in self.bombs:
                    self.screen.blit(self.bomb_image, rect)
                elif pos in self.coins:
                    pygame.draw.circle(
                        self.screen, COIN_COLOR, rect.center, GRID_CELL_SIZE // 2 - 4
                    )
                    font = pygame.font.SysFont(None, 24)
                    text = font.render(str(self.coins[pos]), True, (0, 0, 0))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        p1_rect = pygame.Rect(
            GRID_OFFSET_X + self.agent_positions[0][1] * GRID_CELL_SIZE,
            GRID_OFFSET_Y + self.agent_positions[0][0] * GRID_CELL_SIZE,
            GRID_CELL_SIZE,
            GRID_CELL_SIZE,
        )
        self.screen.blit(self.player1_image, p1_rect)
        p2_rect = pygame.Rect(
            GRID_OFFSET_X + self.agent_positions[1][1] * GRID_CELL_SIZE,
            GRID_OFFSET_Y + self.agent_positions[1][0] * GRID_CELL_SIZE,
            GRID_CELL_SIZE,
            GRID_CELL_SIZE,
        )
        self.screen.blit(self.player2_image, p2_rect)

        pygame.display.flip()
