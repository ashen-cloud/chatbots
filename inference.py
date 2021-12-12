#!/opt/homebrew/bin/python3

import torch

model = torch.load('trained_model', map_location=torch.device('cuda:0' if torch.cuda.is_available() else'cpu'))

print(model)

# query = input()



