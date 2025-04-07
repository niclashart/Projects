import gym
from gym import spaces
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Define a custom environment
class TreasureHuntEnv(gym.Env):
    def __init__(self, config=None):
        super(TreasureHuntEnv, self).__init__()
        self.grid_size = 10
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)  # 2D position
        self.action_space = spaces.Discrete(4)  # 4 actions (up, down, left, right)
        self.state = np.array([0, 0])  # Initial state
        self.treasure = np.array([5, 5])  # Treasure location
        self.steps = 0
        self.max_steps = 50

    def reset(self):
        self.state = np.array([0, 0])
        self.steps = 0
        self.render()  # Render the initial state
        return self.state

    def step(self, action):
        self.steps += 1
        if action == 0:  # Up
            self.state[1] = min(self.state[1] + 1, self.grid_size - 1)
        elif action == 1:  # Down
            self.state[1] = max(self.state[1] - 1, 0)
        elif action == 2:  # Left
            self.state[0] = max(self.state[0] - 1, 0)
        elif action == 3:  # Right
            self.state[0] = min(self.state[0] + 1, self.grid_size - 1)

        # Calculate reward
        done = np.array_equal(self.state, self.treasure) or self.steps >= self.max_steps
        reward = 10 if np.array_equal(self.state, self.treasure) else -0.1

        self.render()  # Render the current state
        return self.state, reward, done, {}

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.treasure[1]][self.treasure[0]] = "T"  # Mark the treasure
        grid[self.state[1]][self.state[0]] = "A"  # Mark the agent
        print("\n".join(" ".join(row) for row in grid))
        print("\n" + "-" * (self.grid_size * 2 - 1))  # Separator for clarity

# Register the custom environment
gym.envs.registration.register(
    id="TreasureHunt-v0",
    entry_point=TreasureHuntEnv,
)

# Initialize Ray
ray.init()

# Define the training configuration
config = (
    PPOConfig()
    .environment("TreasureHunt-v0")  # Use the custom environment
    .framework("torch")              # Use PyTorch as the deep learning framework
    .rollouts(num_rollout_workers=1)  # Number of parallel workers
    .training(train_batch_size=4000, gamma=0.99)  # Training parameters
)

# Train the agent
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"episode_reward_mean": 9},  # Stop when the average reward is close to 10
    checkpoint_at_end=True,           # Save the final checkpoint
)

# Shutdown Ray
ray.shutdown()