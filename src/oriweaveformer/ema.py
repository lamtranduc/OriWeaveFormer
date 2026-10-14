import copy

import torch


class EMA:
    def __init__(self, model, decay):
        self.decay, self.updates = decay, 0
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for target, source in zip(self.shadow.state_dict().values(),
                                  model.state_dict().values()):
            if target.dtype.is_floating_point:
                target.mul_(decay).add_(source.float(), alpha=1 - decay)
            else:
                target.copy_(source)
