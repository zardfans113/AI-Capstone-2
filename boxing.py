# boxing.py
import ale_py
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from dqn_agent import DQN, ReplayBuffer
from ale_py import ALEInterface
import os
import cv2

class BoxingEnv(gym.Env):
    """
    自訂 Boxinig Atari 環境
    只在此定義環境，訓練、評估請放到其他函式或檔案
    """
    def __init__(self):
        super().__init__()
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", 42)
        self.ale.setBool("display_screen", False)

        rom_path = os.path.join(
            os.path.dirname(ale_py.__file__),
            "roms",
            "boxing.bin"
        )
        self.ale.loadROM(rom_path)

        self.actions = self.ale.getLegalActionSet()
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(210, 160, 3),
            dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        self.ale.reset_game()
        frame = self.ale.getScreenRGB()
        return frame, {}

    def step(self, action):
        total_reward = 0
        for _ in range(4):  # Frame skip 4
            reward = self.ale.act(self.actions[action])
            total_reward += reward
            if self.ale.game_over():
                break
        frame = self.ale.getScreenRGB()
        done = self.ale.game_over()
        return frame, total_reward, done, False, {}

    def render(self):
        frame = self.ale.getScreenRGB()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Boxing", bgr)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


def train(num_frames=500_000, save_prefix="q_net"):
    """
    訓練主程式，將程式搬到此函式內，避免直接 import 時自動執行
    num_frames: 總訓練幀數
    save_prefix: 模型檔名前綴
    """
    env = BoxingEnv()
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立網路與 optimizer
    q_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)

    buffer = ReplayBuffer(50_000)
    epsilon = 1.0
    eps_decay = 0.990
    eps_min = 0.05
    gamma = 0.99
    batch_size = 32
    best_reward = float('-inf')
    reward_sum = 0

    state, _ = env.reset()
    for frame in range(1, num_frames+1):
        # Epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            tensor_state = torch.tensor(
                state.transpose(2,0,1)[None],
                dtype=torch.float32, device=device
            )
            action = q_net(tensor_state).argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        reward *= 10  # 放大 reward
        buffer.push(state, action, reward, next_state, done)
        reward_sum += reward
        state = next_state

        # 一局結束後處理
        if done:
            if reward_sum > best_reward:
                best_reward = reward_sum
                torch.save(q_net.state_dict(), f"{save_prefix}_best.pth")
                print(f"🏆 新最佳 reward: {best_reward:.2f}, 已存檔 {save_prefix}_best.pth")
            reward_sum = 0
            state, _ = env.reset()
            epsilon = max(eps_min, epsilon * eps_decay)

        # 訓練更新
        if len(buffer) >= batch_size:
            s, a, r, s_, d = buffer.sample(batch_size)
            s = torch.tensor(s.transpose(0,3,1,2), dtype=torch.float32, device=device)
            a = torch.tensor(a, device=device)
            r = torch.tensor(r, dtype=torch.float32, device=device)
            s_ = torch.tensor(s_.transpose(0,3,1,2), dtype=torch.float32, device=device)
            d = torch.tensor(d, dtype=torch.float32, device=device)

            q_val = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
            q_next = target_net(s_).max(1)[0]
            q_target = r + gamma * q_next * (1 - d)
            loss = (q_val - q_target.detach()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 定期同步 target_net
        if frame % 1000 == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f"Frame {frame} | Epsilon {epsilon:.3f} | Loss {loss.item():.4f}")

        # 定期存檔 checkpoint
        if frame % 100_000 == 0:
            torch.save(q_net.state_dict(), f"{save_prefix}_{frame}.pth")
            print(f"💾 已儲存 checkpoint: {save_prefix}_{frame}.pth")

    # 訓練結束存最終模型
    torch.save(q_net.state_dict(), f"{save_prefix}_final.pth")
    print(f"✅ 訓練完成，最終模型已儲存為 {save_prefix}_final.pth")


if __name__ == "__main__":
    # 只有直接執行 boxing.py 時才會進行訓練
    train()
