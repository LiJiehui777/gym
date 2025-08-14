import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import gym
import time, random
from loguru import logger


class PPOModelWrapper:
    def __init__(self, model_path, device="auto"):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )

        self.model = self._load_model(model_path)

    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        class WrappedAgent(nn.Module):
            def __init__(self):
                super().__init__()
                # Network部分（堆叠三帧）
                self.network = nn.Sequential(
                    self.layer_init(nn.Conv2d(15, 32, 3, stride=1)),
                    nn.ReLU(),
                    self.layer_init(nn.Conv2d(32, 64, 3, stride=1)),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Flatten(),
                    self.layer_init(nn.Linear(64 * 2 * 2, 128)),
                    nn.ReLU(),
                )
                # Actor部分
                self.actor = nn.Sequential(
                    self.layer_init(nn.Linear(128, 64)),
                    nn.ReLU(),
                    self.layer_init(nn.Linear(64, 32)),
                    nn.ReLU(),
                    self.layer_init(nn.Linear(32, 5), std=0.01),
                )

                self.critic = nn.Sequential(
                    self.layer_init(nn.Linear(128, 64)),
                    nn.ReLU(),
                    self.layer_init(nn.Linear(64, 16)),
                    nn.ReLU(),
                    self.layer_init(nn.Linear(16, 1), std=1),
                )

            def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
                torch.nn.init.orthogonal_(layer.weight, std)
                torch.nn.init.constant_(layer.bias, bias_const)
                return layer

        model = WrappedAgent()
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()
        return model

    def predict(self, observation):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            hidden = self.model.network(obs_tensor)
            logits = self.model.actor(hidden)
            probs = Categorical(logits=logits)
            best_action = probs.sample().item()
            # best_action = torch.argmax(probs.probs).item()
            return best_action


class Player:
    def __init__(self):
        self.gold1 = 0
        self.gold2 = 0
        self.model = None
        self.last_location = None
        self.stack_frames = 3
        self.historical_states = None
        if model_path:
            self.model = PPOModelWrapper('/srv/nfs/share/users/Ubiquant78/checkpoint_step19200000.pt')

    def MoveDecision(self, grid: list[list[int]], gold: int, gold_2: int) -> list[int]:
        self.gold1 = gold
        self.gold2 = gold_2

        if self.model:
            return self._model_predict(grid)
        return [random.randint(0, 3) for _ in range(3)]

    def _model_predict(self, grid):
        observation = self._preprocess_grid(grid)
        if self.historical_states == None:
            processed_obs = self._normalize_observation(observation)
            self.historical_states = [processed_obs for _ in range(self.stack_frames)]

        result = []
        current_obs = np.copy(observation)

        for _ in range(3):
            # 模型预测最佳动作
            best_action = self.model.predict(np.vstack(self.historical_states))
            result.append(best_action)

            # 模拟环境状态变化
            current_obs = self._simulate_action(current_obs, best_action, grid)
            # if isinstance(current_obs, np.ndarray):
            #     np.set_printoptions(threshold=np.inf)
            #     print("next_obs shape:", current_obs.shape)
            #     print(np.vstack(self.historical_states).shape)

            # 预处理当前观察值
            processed_obs = self._normalize_observation(current_obs)
            self.historical_states.append(processed_obs)
            self.historical_states.pop(0)

        return result

    def _simulate_action(self, obs, action, grid):
        new_obs = np.copy(obs)
        r, c = np.where(obs[0] == 1)
        r, c = r.item(), c.item()
        r_new, c_new = r, c

        if action == 0 and r > 0 and grid[r - 1][c] != -2 and new_obs[2, r - 1, c] != 1:
            r_new, c_new = r - 1, c  # 上移
        elif (
            action == 1
            and r < 16
            and grid[r + 1][c] != -2
            and new_obs[2, r + 1, c] != 1
        ):  # 下移
            r_new, c_new = r + 1, c
        elif (
            action == 2 and c > 0 and grid[r][c - 1] != -2 and new_obs[2, r, c - 1] != 1
        ):  # 左移
            r_new, c_new = r, c - 1
        elif (
            action == 3
            and c < 16
            and grid[r][c + 1] != -2
            and new_obs[2, r, c + 1] != 1
        ):  # 右移
            r_new, c_new = r, c + 1

        new_obs[0, r, c], new_obs[0, r_new, c_new] = 0, 1

        # 新位置是金币
        if new_obs[1, r_new, c_new] >= 1:
            new_obs[1, r_new, c_new] = 0
        # 新位置是炸弹
        elif new_obs[3, r_new, c_new] == 1:
            new_obs[3, r_new, c_new] = 0

        # 处理上一步的动作
        if self.last_location:
            r1, c1 = self.last_location
            new_obs[4, r1, c1] = 0
        new_obs[4, r, c] = action + 1
        self.last_location = [r, c]

        return new_obs

    def _normalize_observation(self, obs):
        normalized = np.copy(obs)
        normalized[1] /= 25.0  # 金币值归一化
        normalized[4] /= 5.0  # 历史动作归一化
        return normalized

    def _preprocess_grid(self, grid):
        # 原始输入格式：[5, 17, 17]的numpy数组
        # Channel 0: Agent位置
        # Channel 1: 金币
        # Channel 2: 障碍物
        # Channel 3: 炸弹
        # Channel 4: 上一步动作

        obs = np.zeros((5, 17, 17), dtype=np.float32)

        for r in range(17):
            for c in range(17):
                val = grid[r][c]
                if val == -9:  # 玩家自己
                    obs[0, r, c] = 1
                elif val == -2:  # 对手
                    pass
                elif val == -1:  # 障碍物
                    obs[2, r, c] = 1
                elif val == -3:  # 炸弹
                    obs[3, r, c] = 1
                elif val >= 1:  # 金币
                    obs[1, r, c] = val

        return obs


