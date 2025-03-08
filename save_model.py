import os
import torch
import torch.nn as nn

# ✅ Ensure the directory exists
MODEL_PATH = r"C:\Users\Ryan\bob_ai\core\data\trained_model.pth"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ✅ Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# ✅ Initialize and save the model
model = SimpleModel()
torch.save(model.state_dict(), MODEL_PATH)

# ✅ Confirm model is saved
if os.path.exists(MODEL_PATH):
    print(f"✅ Model successfully saved at: {MODEL_PATH}")
else:
    print("❌ ERROR: Model was NOT saved! Check directory permissions.")
