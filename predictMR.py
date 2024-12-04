import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(5, 64),
            nn.Sigmoid(),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.model(x)
        return x

# Utility to load model checkpoint
def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']

# CPU-optimized forward pass
@torch.no_grad()
def cpu_forward_pass(inputs):
    model.eval()
    inputs = inputs.to(device)
    predictions = scripted_model(inputs)  # Use scripted model for optimized inference
    return predictions.cpu().numpy()

# Generic forward pass
@torch.no_grad()
def generic_forward_pass(dataloader):
    model.eval()
    predictions = []
    for X in dataloader:
        X = X.to(device)
        pred = model(X)
        predictions.append(pred.cpu().numpy())
    return np.vstack(predictions)

# CPU-optimized MRpredict
def cpu_MRpredict(pc, p0, G1, G2, G3):
    Xinput = np.array([[pc, p0, G1, G2, G3]], dtype=np.float32)
    Xinput = scaler_x.transform(Xinput)
    test_data = torch.tensor(Xinput, dtype=torch.float32).to(device)
    predicted = cpu_forward_pass(test_data)
    return scaler_y.inverse_transform(predicted)[0]

# Generic MRpredict
def generic_MRpredict(pc, p0, G1, G2, G3):
    Xinput = np.array([[pc, p0, G1, G2, G3]])
    Xinput = scaler_x.transform(Xinput)
    test_data = torch.tensor(Xinput, dtype=torch.float32).to(device)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    predicted = generic_forward_pass(test_loader)
    return scaler_y.inverse_transform(predicted)[0]

# CPU-optimized predict_MR_curve
def cpu_predict_MR_curve(p0, G1, G2, G3, with_pc = False):
    Pc_array = np.logspace(np.log(5), np.log(np.random.uniform(800, 950)), num=100, base=np.e)
    pc = np.log(Pc_array)
    Xinput = np.column_stack([pc, np.full_like(pc, p0), np.full_like(pc, G1),
                              np.full_like(pc, G2), np.full_like(pc, G3)])
    Xinput = scaler_x.transform(Xinput)
    test_data = torch.tensor(Xinput, dtype=torch.float32).to(device)
    predicted = cpu_forward_pass(test_data)
    MR_array = scaler_y.inverse_transform(predicted)
    if with_pc == True:
        out = pd.DataFrame(MR_array,Pc_array)
        out[len(out.columns)] = Pc_array
    else:
        out = pd.DataFrame(MR_array)
    return out

# Generic predict_MR_curve
def generic_predict_MR_curve(p0, G1, G2, G3,with_pc = False):
    Pc_array = np.logspace(np.log(5), np.log(np.random.uniform(800, 950)), num=100, base=np.e)
    pc = np.log(Pc_array)
    MR_array = [generic_MRpredict(p, p0, G1, G2, G3) for p in pc]
    if with_pc == True:
        out = pd.DataFrame(MR_array,Pc_array)
        out[len(out.columns)] = Pc_array
    else:
        out = pd.DataFrame(MR_array)
    return out

# Initialize the model
scaler_x = joblib.load('scalerX.gz')
scaler_y = joblib.load('scalery.gz')

model = NeuralNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
ckp_path = "NNmodel_GR.ckp"
model, optimizer, start_epoch = load_checkpoint(model, optimizer, ckp_path)

# Convert model to TorchScript if CPU is used
scripted_model = None
if device.type == "cpu":
    scripted_model = torch.jit.script(model)

# Define which version of functions to use based on device
if device.type == "cpu":
    forward_pass = cpu_forward_pass
    MRpredict = cpu_MRpredict
    predict_MR_curve = cpu_predict_MR_curve
else:
    forward_pass = generic_forward_pass
    MRpredict = generic_MRpredict
    predict_MR_curve = generic_predict_MR_curve

# Example usage
# SLy_param = (p0, G1, G2, G3)
# massrad = predict_MR_curve(*SLy_param)
