import torch

state_dict = torch.load('src/pages/model.pth', map_location=torch.device('cpu'))

for key, value in state_dict.items():
    print(f"{key}: {value.shape}")