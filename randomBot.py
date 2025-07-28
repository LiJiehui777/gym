import random


class Player:
    def __init__(self):
        self.gold = 0

    # grid {0: 空白，-9: 玩家本人，-1: 障碍物，-2: 对手玩家，-3: 炸弹，>=1的正整数: 金币量}
    # 返回值: { 0:上，1:下，2:左，3:右 }
    # gold：自己目前金币数量，gold_2：对手目前金币数量
    def MoveDecision(self, grid: list[list[int]], gold: int, gold_2: int) -> list[int]:
        # 这里可以添加更复杂的决策逻辑
        return [random.randint(0, 4) for _ in range(3)]
