import importlib.util
import os
from typing import List
from pathlib import Path
import pdb

module_path = (
    Path(__file__).parent.joinpath("build/micro.cpython-310-darwin.so").resolve()
)
# pdb.set_trace()
spec = importlib.util.spec_from_file_location("micro", module_path)
micro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(micro)


class Player:
    def __init__(self):
        self.player = micro.Player()

    def MoveDecision(self, grid: List[List[int]], gold: int, gold2: int) -> list:
        return self.player.MoveDecision(grid, gold, gold2)
