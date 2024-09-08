import torch

class MyFC(object):
    def __init__(self, inputs_size, outputs_size, device):
        self.W = torch.rand(size=(inputs_size, outputs_size), device=device)
        self.W.requires_grad_(True)
        self.b = torch.zeros(outputs_size, device=device)
        self.b.requires_grad_(True)

    def forward(self, inputs):
        out = torch.empty((inputs.shape[0], 1))
        count = 0
        for x in inputs:
            out[count] = (x @ self.W) + self.b
            count += 1
        out = torch.softmax(out, dim=0)
        return out