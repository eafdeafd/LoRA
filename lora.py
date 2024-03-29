import torch
import torch.nn as nn
import torch.functional as F

#class LoRALayer(nn.Module):
#    def __init__(self, in_dim, out_dim, rank, alpha):
#        super().__init__()
#        std_dev = 1 / torch.sqrt(torch.tensor(rank))
#        self.rank = rank
#        self.alpha = alpha
#        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
#        self.B = nn.Parameter(torch.zeros(rank, out_dim))
#    
#    def forward(self, x):
#        return self.alpha * (x @ self.A @ self.B)

class LoRALayer(nn.Module):
    # the same as lora, just conceptually nicer imo, can also add bias
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.seq = nn.Sequential(
            nn.Linear(in_dim, rank, bias=False),
            nn.Linear(rank, out_dim, bias=False)
        )
        self.seq[0].weight.data = torch.randn(rank, in_dim) * (1 / torch.sqrt(torch.tensor(rank)))
        self.seq[1].weight.data = torch.zeros(out_dim, rank)
    
    def forward(self, x):
        return self.alpha * self.seq(x)
    
class LinearWithLoRA(torch.nn.Module):
    # wrapper around Linear Layers for normal LoRA
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
class ProgressiveLoRANet(nn.Module):
    # progressive net style - but only one connection vertically
    # TODO: test and add diag connections
    def __init__(self, base_model, lora_rank, lora_alpha):
        super().__init__()
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_layers = nn.ModuleList()
    
    def add_lora_layer(self):
        lora_layer = LoRALayer(self.base_model.out_features, self.base_model.out_features, self.lora_rank, self.lora_alpha)
        self.lora_layers.append(lora_layer)
    
    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.base_model(x)
        for lora_layer in self.lora_layers:
            x = x + lora_layer(x)
        return x
    
class LinearWithDoRA(torch.nn.Module):
    # Code inspired by https://github.com/catid/dora/blob/main/dora.py
    # from LightningAI (see 02_finetune-with-lora.ipynb)
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha*lora.T
        column_norm = combined_weight.norm(p=2, dim=0, keepdim=True)
        V = combined_weight / column_norm
        new_weight = self.m * V
        return F.linear(x, new_weight, self.linear.bias)