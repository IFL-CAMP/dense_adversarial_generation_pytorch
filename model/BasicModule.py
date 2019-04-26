import torch
import torch.nn as nn
import time

class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if not name:
            prefix = 'checkpoints/' + 'bay' +self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H%M.pkl')
        torch.save(self.state_dict(), name)
        return name
