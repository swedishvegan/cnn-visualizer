import torch
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = models.googlenet(weights = "GoogLeNet_Weights.IMAGENET1K_V1").to(device)

for param in _model.parameters(): param.requires_grad_(False)
_model.eval()
_model.transform_input = False

class Model(torch.nn.Module):
    
    def __init__(self): super().__init__()
    
    def forward(self, x, end_layer = "", features = None):
        
        for m in _model._modules: 
        
            if _model._modules[m] is None: continue
            if m == "fc": x = torch.flatten(x, 1)
            
            x = _model._modules[m](x)
            if m == end_layer and end_layer != "fc": return x if features is None else x[:, features, :, :]
            
        return x if features is None else x[:, features]
    
_test = torch.randn(1, 3, 224, 224).to(device)
assert torch.sum((torch.abs(Model()(_test) - _model(_test))).to("cpu")).item() < 0.001