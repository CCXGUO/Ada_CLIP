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
        self.dir1 = 'C:/Users/17839/dev/Data/AdaClip/data/Video1'
        self.dir2 = 'C:/Users/17839/dev/Data/AdaClip/data/Video2'

config = Config()
