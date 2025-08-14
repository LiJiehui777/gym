# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

# import gymnasium as gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from pathlib import Path
import shutil


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "GoldRushNew-v0"
    """the id of the environment"""
    total_timesteps: int = 100000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.80
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            # 第一层卷积：5×17×17 → 32×15×15（无padding）
            # layer_init(nn.Conv2d(12, 32, 3, stride=1)),
            layer_init(nn.Conv2d(12, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            # 第二层卷积：32×15×15 → 64×13×13（无padding）
            layer_init(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            # 第一次池化：64×13×13 → 64×6×6（2×2池化）
            nn.MaxPool2d(2, 2),
            # 第三层卷积：64×6×6 → 64×4×4（无padding）
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            # 第二次池化（可选，进一步压缩尺寸）：64×4×4 → 64×2×2
            nn.MaxPool2d(2, 2),
            # 展平：64×2×2 = 256
            nn.Flatten(),
            # 修正线性层输入维度（256 → 256）
            layer_init(nn.Linear(64 * 3 * 3, 17 * 17)),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(17 * 17 * 4, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(17 * 17 * 4, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128 )),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1)
        )
        # self.agent_location = nn.Sequential(
        #     layer_init(nn.Linear(3 * 17 * 17, 128)),
        #     nn.ReLU()
        # )


    def get_action_and_value(self, x, action=None):
        # 提取第0、5、10通道
        selected_indices = [0, 5, 10]
        agent_obs = x[:, selected_indices, :, :].reshape(-1, 3 * 17 * 17)
        mask = np.ones(x.shape[1], dtype=bool)  
        mask[selected_indices] = False            
        x = x[:, mask, :, :]

        hidden = self.network(x)
        # hidden_agent_location = self.agent_location(agent_obs)
        hidden_agent_location = agent_obs
        hidden = torch.cat([hidden, hidden_agent_location], axis=1)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


    def get_value(self, x):
        selected_indices = [0, 5, 10]
        agent_obs = x[:, selected_indices, :, :].reshape(-1, 3 * 17 * 17)
        mask = np.ones(x.shape[1], dtype=bool)  
        mask[selected_indices] = False            
        x = x[:, mask, :, :]

        hidden = self.network(x)
        # hidden_agent_location = self.agent_location(agent_obs)
        hidden_agent_location = agent_obs
        hidden = torch.cat([hidden, hidden_agent_location], axis=1)
        return self.critic(hidden)

    # def get_action_and_value(self, x, action=None):
    #     hidden = self.network(x)
    #     logits = self.actor(hidden)
    #     probs = Categorical(logits=logits)
    #     if action is None:
    #         action = probs.sample()
    #     return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    # print(gym.__file__)
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)  # 128
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 32
    args.num_iterations = args.total_timesteps // args.batch_size  # 78125
    run_name = f"256_sqrt_newest_steps_to_eat_coins_{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # 创建模型保存目录
    model_dir = Path(f"runs/{run_name}/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 初始化最佳成绩记录
    best_score = -float('inf')  # 用于保存最佳模型
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    # envs = gym.make(args.env_id)
    # envs = gym.make(args.env_id, maze="maze1")
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # actions = torch.zeros(args.num_steps, args.num_envs, int(envs.single_action_space.nvec[0])).to(device)
    # actions = torch.zeros(args.num_steps, args.num_envs, envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    # if isinstance(next_obs, np.ndarray):
    #     np.set_printoptions(threshold=np.inf)  
    #     print("next_obs shape:", next_obs.shape)
    #     print(next_obs)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):  # 128
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # 假设当前只有一个玩家
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # player1_action = tuple([(0, action.cpu().numpy().item())] * Args.num_envs)
            # next_obs, reward, terminations, truncations, infos = envs.step(player1_action)
            # next_obs = _next_obs.squeeze(axis=0)

            # if isinstance(next_obs, np.ndarray):
            #     with open("output.txt", "a") as f:
            #         np.set_printoptions(threshold=np.inf)  # 防止截断
            #         f.write(f"action: {action}\n")
            #         f.write(f"next_obs shape: {next_obs.shape}\n")
            #         f.write(np.array2string(next_obs, threshold=np.inf))
            # if isinstance(next_obs, np.ndarray):
            #     np.set_printoptions(threshold=np.inf)  
            #     print("next_obs shape:", next_obs.shape)
            #     print(next_obs)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, ))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step % 100000 == 0:
            checkpoint_path = model_dir / f"checkpoint_step{global_step}.pt"
            torch.save({
                'model': agent.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'global_step': global_step,
                'timestamp': time.time()
            }, checkpoint_path)
            print(f"Checkpoint saved at step {global_step}")

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and 'episode' in info:
                    current_score = info['episode']['r']
                    if current_score > best_score:
                        best_score = current_score
                        best_model_path = model_dir / "best_model.pt"
                        torch.save(agent.state_dict(), best_model_path)
                        print(f"New best model saved with score: {best_score:.2f}")

    final_model_path = model_dir / "final_model.pt"
    torch.save({
        'model': agent.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(args),
        'total_steps': global_step,
        'final_score': best_score,
        'training_time': time.time() - start_time
    }, final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

    envs.close()
    writer.close()
