import torch
class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 32
        self.num_epochs = 25
        self.learning_rate = 0.001
        self.threshold = 0.5
        self.sigma = 1.0
        self.weights_save_path = 'model_weights'
        self.dir1 = 'data/Video1'
        self.dir2 = 'data/Video2'

config = Config()
