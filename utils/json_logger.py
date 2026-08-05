import json
import os
import datetime

class JSONLogger:
    def __init__(self, path, name):
        """
        Args:
            path (str): The directory where the log file will be saved.
            name (str): The name of the log file (usually the unique_token of the run).
        """
        self.log_path = os.path.join(path, f"{name}.json")
        self.log_data = {}

        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

    def log_stat(self, key, value, t_env):
        """
        Logs a single key-value pair statistic at a given timestep.

        Args:
            key (str): The name of the statistic (e.g., "test_return_mean").
            value (float or int): The value of the statistic.
            t_env (int): The environment timestep at which the statistic was recorded.
        """
        if key not in self.log_data:
            self.log_data[key] = []
        
        self.log_data[key].append({
            "t_env": t_env,
            "value": value
        })

    def save(self):
        """
        Saves the accumulated log data to the JSON file.
        """
        with open(self.log_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)
