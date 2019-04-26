import logging
import torch.nn as nn
import torch
import numpy as np
import time

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name = str(type(self))
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if not name:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H%M.pkl')
        torch.save(self.state_dict(), name)
        return name
