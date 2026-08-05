class TxtLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        try:
            self.file = open(self.filepath, 'w')
        except IOError as e:
            print(f"Error opening log file {self.filepath}: {e}")

    def log(self, t_env, mean_return):
        if self.file:
            self.file.write(f"{t_env},{mean_return}\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
