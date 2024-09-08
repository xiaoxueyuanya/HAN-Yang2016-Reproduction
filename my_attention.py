import torch

class MyAttention(object):
    def __init__(self, inputs_size, device):
        params_w = inputs_size        
        self.W_w = torch.randn(size=(params_w, params_w), device=device) * 0.01
        self.W_w.requires_grad_(True)
        self.b_w = torch.zeros(params_w, device=device)
        self.b_w.requires_grad_(True)
        self.u_w = torch.rand(size=(params_w, 1), device=device)
        self.u_w.requires_grad_(True)

    def forward(self, inputs):
        alpha_i = torch.empty((inputs.shape[0], 1))
        count = 0
        for x in inputs:
            u_it = torch.tanh((x @ self.W_w) + self.b_w)
            alpha_i[count] = (u_it @ self.u_w)
            count += 1
        alpha_i = torch.softmax(alpha_i, dim=0)
            
        s = (alpha_i * inputs).sum(dim=0)
        return s