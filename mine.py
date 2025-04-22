import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import imageio
from tqdm import tqdm
import random
import numpy as np


# 參數設定
GRID_SIZE = 9
NUM_MINES = 10
TILE_SIZE = 32
WINDOW_SIZE = GRID_SIZE * TILE_SIZE
MAX_STEPS = 100

# 定義 Minesweeper 環境
class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.observation_space = spaces.Box(low=0, high=10, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.action_space = spaces.Discrete(GRID_SIZE * GRID_SIZE)

        self._reset_board()
    
    def _reset_board(self):
        if not hasattr(self, 'initial_board'):
            # 第一次 reset：隨機生成
            self.revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            self.board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
            self.mines = set()

            while len(self.mines) < NUM_MINES:
                x, y = np.random.randint(0, GRID_SIZE, size=2)
                self.mines.add((x, y))

            for x, y in self.mines:
                self.board[x, y] = -1
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.board[nx, ny] != -1:
                            self.board[nx, ny] += 1

            # 儲存初始雷盤
            self.initial_board = self.board.copy()
            self.initial_mines = self.mines.copy()
        else:
            # 之後的 reset：回到初始狀態
            self.board = self.initial_board.copy()
            self.mines = self.initial_mines.copy()
            self.revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

        # 更新未翻開的位置
        self.available_positions = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]

    def _reset_board_rand(self):
        self.revealed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        self.board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.mines = set()

        while len(self.mines) < NUM_MINES:
            x, y = np.random.randint(0, GRID_SIZE, size=2)
            self.mines.add((x, y))

        for x, y in self.mines:
            self.board[x, y] = -1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.board[nx, ny] != -1:
                        self.board[nx, ny] += 1
        
        self.available_positions = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]  # 所有未翻開的位置

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_board()
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.where(self.revealed, self.board, 10)  # 10 代表未翻開
        return obs

    def _get_adjacent_mines2(self, x, y):
        # 計算該格子周圍的地雷數量
        mines = 0
        total = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx * dy == -1:
                    x1, y1 = x + dx, y + dy
                    x2, y2 = x + dx, y
                    x3, y3 = x, y + dy
                    check = 3
                    if 0 <= x1 < GRID_SIZE and 0 <= y1 < GRID_SIZE and (self.revealed[x1,y1] and ~self.board[x1, y1]):
                        check -= 1
                    if 0 <= x2 < GRID_SIZE and 0 <= y2 < GRID_SIZE and (self.revealed[x2,y2] and ~self.board[x2, y2]):
                        check -= 1
                    if 0 <= x3 < GRID_SIZE and 0 <= y3 < GRID_SIZE and (self.revealed[x3,y3] and ~self.board[x3, y3]):
                        check -= 1
                    if check == 3:
                        return 1.0
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    total += 1
                    if self.revealed[nx, ny] and self.board[nx, ny] > 0:
                        mines += 1

        return mines / total
    
    def _get_adjacent_mines(self, x, y):
        # 計算該格子周圍的地雷數量
        mines = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if self.revealed[nx, ny] and self.board[nx, ny] > 0:
                        mines += 1

        return mines

    def step(self, action):
        x, y = divmod(action, GRID_SIZE)

        board = self._reveal_tile(x, y)

        # 如果是地雷，立刻結束
        if (x, y) in self.mines:
            return self._get_obs(), -15.0, True, False, {}

        # 這裡可以根據鄰近地雷數量設計更細緻的回報策略
        reward = board * 0.5 + 8.0 - self.board[x, y] # 無地雷，翻開格子獲得較高的回報

        if np.all((self.board != -1) == self.revealed):
            return self._get_obs(), 20.0, True, False, {}

        return self._get_obs(), reward, False, False, {}


    def _reveal_tile(self, x, y):
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) or self.revealed[x, y]:
            return 0
        total = 1
        self.revealed[x, y] = True
        self.available_positions.remove((x, y))  # 移除已翻開的位置
        if self.board[x, y] == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        total += self._reveal_tile(x + dx, y + dy)
        return total
                

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            self.clock = pygame.time.Clock()

        self.window.fill((200, 200, 200))
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if self.revealed[x, y]:
                    value = self.board[x, y]

                    if value == -1:
                        # 地雷格子背景紅色，文字為 *
                        pygame.draw.rect(self.window, (255, 0, 0), rect)
                        font = pygame.font.SysFont(None, 24)
                        text = font.render("*", True, (0, 0, 0))
                        self.window.blit(text, rect.topleft)
                    else:
                        # 普通格子
                        color = (255, 255, 255) if value == 0 else (180, 180, 255)
                        pygame.draw.rect(self.window, color, rect)
                        if value > 0:
                            font = pygame.font.SysFont(None, 24)
                            text = font.render(str(value), True, (0, 0, 0))
                            self.window.blit(text, rect.topleft)

                else:
                    pygame.draw.rect(self.window, (100, 100, 100), rect)
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window).swapaxes(0, 1)

    def close(self):
        if self.window:
            pygame.quit()

    def get_available_position(self):
        # 返回一個尚未翻開的隨機位置
        return random.choice(self.available_positions) if self.available_positions else None

# 定義 DQN 模型
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x.float())

# 定義 DQN Agent
class DQNAgent:
    def __init__(self, state_shape, action_dim, device="cpu"):
        self.device = device
        self.model = DQN(np.prod(state_shape), action_dim).to(device)
        self.target = DQN(np.prod(state_shape), action_dim).to(device)
        self.target.load_state_dict(self.model.state_dict())

        self.buffer = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 100

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.model.net[-1].out_features - 1)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.model(state)
        return torch.argmax(q).item()

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def train(self, step):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = zip(*samples)

        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.array(a), dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(np.array(r), dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(np.array(d), dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.model(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(s2).max(1, keepdim=True)[0]
            q_target = r + self.gamma * q_next * (1 - d)

        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % self.update_target_every == 0:
            self.target.load_state_dict(self.model.state_dict())


def train():
    env = MinesweeperEnv(render_mode="rgb_array")
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    
    epsilon = 1.0
    decay = 0.995
    total_episodes = 2000
    epoch = 1
    for ep in range(epoch):
        print(f"Now is epoch {ep + 1}/{epoch} :")
        with tqdm(total=total_episodes, desc="Training", ncols=120) as pbar:
            for episode in range(total_episodes):
                state, _ = env.reset()
                done = False
                total_reward = 0
                steps = 0

                while not done and steps < MAX_STEPS:
                    flat_state = state.flatten()
                    unrevealed_actions = [i for i, val in enumerate(flat_state) if val == 10]  # 10 = 未翻
                    if unrevealed_actions:
                        # 讓 agent 在這些位置中選擇最大 Q 值
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        q_values = agent.model(state_tensor).detach().numpy()[0]

                        for action in unrevealed_actions:
                            x, y = divmod(action, GRID_SIZE)  # 將一維的 action 映射回 (x, y)
                            adjacent_mines = env._get_adjacent_mines(x, y)  # 計算周圍地雷數量
                            q_values[action] -= adjacent_mines * 2  # 用係數 2 來降低危險地雷格子的 Q-value

                        # 選擇最佳 Q 值對應的 action
                        best_action = max(unrevealed_actions, key=lambda a: q_values[a])
                        action = best_action
                    else:
                        action = agent.act(state, epsilon=0.0)

                    next_state, reward, done, truncated, info = env.step(action)
                    agent.push(state, action, reward, next_state, done)
                    agent.train(steps)
                    state = next_state
                    total_reward += reward
                    steps += 1

                epsilon *= decay
                pbar.set_postfix(episode=episode, reward=total_reward)
                pbar.update(1)

    torch.save(agent.model.state_dict(), "minesweeper_dqn.pth")


# 測試過程並錄製影片
def test():
    best_total = -float('inf')  # 用來記錄最佳的回報
    best_frames = []  # 用來記錄最佳回報的幀
    best_episode = 0  # 記錄最佳測試的編號

    # 進行10次測試
    for episode in range(10):
        env = MinesweeperEnv(render_mode="rgb_array")
        agent = DQNAgent(env.observation_space.shape, env.action_space.n)
        agent.model.load_state_dict(torch.load("minesweeper_dqn.pth"))

        state, _ = env.reset()
        done = False
        frames = []
        total = 0.0

        frame = env.render()
        frames.append(frame)

        while not done:
            # 找出還沒翻開的位置
            flat_state = state.flatten()
            unrevealed_actions = [i for i, val in enumerate(flat_state) if val == 10]

            if unrevealed_actions:
                # 讓 agent 在這些位置中選擇最大 Q 值
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = agent.model(state_tensor).detach().numpy()[0]
                
                for action in unrevealed_actions:
                    x, y = divmod(action, GRID_SIZE)  # 將一維的 action 映射回 (x, y)
                    adjacent_mines = env._get_adjacent_mines(x, y)  # 計算周圍地雷數量
                    q_values[action] -= adjacent_mines * 2  # 用係數 2 來降低危險地雷格子的 Q-value
                
                # 只考慮未翻格子中的 Q 值
                best_action = max(unrevealed_actions, key=lambda a: q_values[a])
                action = best_action
            else:
                action = agent.act(state, epsilon=0.0)

            state, reward, done, truncated, info = env.step(action)
            total += reward
            frame = env.render()
            frames.append(frame)

        print(f"Episode {episode+1} Reward = {total}")

        # 如果這是目前的最佳結果，則保存該結果
        if total > best_total:
            best_total = total
            best_frames = frames
            best_episode = episode + 1

        env.close()

    print(f"Best Episode = {best_episode}, Best Reward = {best_total}")

    # 輸出最佳結果的影片
    writer = imageio.get_writer("minesweeper_best_test.mp4", fps=1)
    for frame in best_frames:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    train()
    test()