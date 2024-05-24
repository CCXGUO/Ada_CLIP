import torch
class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dir1 = 'path_to_dir1'
        self.dir2 = 'path_to_dir2'
        self.batch_size = 32
        self.num_epochs = 25
        self.learning_rate = 0.001
        self.threshold = 0.5
        self.sigma = 1.0
        self.weights_save_path = 'path_to_weights'

config = Config()
