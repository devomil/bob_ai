import os
import torch
import torch.nn as nn

# ✅ Define the correct model path
MODEL_PATH = r"C:\Users\Ryan\bob_ai\core\data\trained_model.pth"

# ✅ Ensure the model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

# ✅ Define SimpleModel (Same as used in save_model.py)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# ✅ Initialize model
model = SimpleModel()

# ✅ Load trained weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"✅ AI Model Successfully Loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR: Failed to load model - {e}")

# ✅ Test model with dummy input
dummy_input = torch.randn(1, 10)  # Match input size
output = model(dummy_input)
print(f"📊 Model Output: {output}")
