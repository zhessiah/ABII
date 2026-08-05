import os
import json
import numpy as np

class RewardTracker:
    def __init__(self, experiment_name, save_dir="data"):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.train_rewards = []
        self.test_rewards = []
        self.train_steps = []
        self.test_steps = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def add_train_reward(self, reward, step):
        """Add a training episode reward and corresponding step."""
        self.train_rewards.append(float(reward))
        self.train_steps.append(int(step))
        self._save_data()
        
    def add_test_reward(self, reward, step):
        """Add a test episode reward and corresponding step."""
        self.test_rewards.append(float(reward))
        self.test_steps.append(int(step))
        self._save_data()
        
    def _save_data(self):
        """Save the reward data to a JSON file."""
        data = {
            "train_rewards": self.train_rewards,
            "train_steps": self.train_steps,
            "test_rewards": self.test_rewards,
            "test_steps": self.test_steps
        }
        
        filename = os.path.join(self.save_dir, f"{self.experiment_name}_rewards.json")
        with open(filename, 'w') as f:
            json.dump(data, f)
