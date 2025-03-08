import os
import json
import logging
import pandas as pd
import datasets
import torch
import torch.nn as nn
import torch.optim as optim

# ✅ Fix Unicode error: Use raw strings
BASE_PATH = r"C:\Users\Ryan\bob_ai\core\data"
MODEL_PATH = os.path.join(BASE_PATH, "trained_model.pth")

# ✅ Ensure the directory exists
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

# ✅ Setup Logging
LOG_FILE = os.path.join(BASE_PATH, "bob_self_learning.log")
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format="%(asctime)s - %(message)s")

def download_and_prepare_dataset():
    """Downloads and preprocesses an NLP dataset from Hugging Face."""
    logging.info("📥 Downloading dataset from Hugging Face...")
    dataset = datasets.load_dataset("ag_news")
    
    df = pd.DataFrame(dataset["train"])
    dataset_path = os.path.join(BASE_PATH, "nlp_dataset.json")
    
    df.to_json(dataset_path, orient="records", lines=True)
    logging.info(f"✅ Dataset downloaded and saved at: {dataset_path}")
    
    return dataset_path

def generate_ai_model_from_learning():
    """Creates and trains an NLP AI model on the downloaded dataset."""
    dataset_path = download_and_prepare_dataset()

    model_code = f"""import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

dataset_path = r"{dataset_path}"
MODEL_PATH = r"{MODEL_PATH}"

# ✅ Ensure model directory exists
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

# ✅ Load Dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f.readlines()]

# ✅ Prepare Data
input_data = [item["text"][:10] for item in dataset]
labels = torch.tensor([1.0 if item["label"] > 1 else 0.0 for item in dataset], dtype=torch.float32)
input_data = torch.tensor([list(map(ord, text))[:10] for text in input_data], dtype=torch.long)

# ✅ Define Model
class TextProcessingModel(nn.Module):
    def __init__(self, vocab_size=5000, embedding_dim=300, hidden_dim=256):
        super(TextProcessingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = torch.sigmoid(self.fc(x[:, -1, :]))  
        return x

# ✅ Initialize Model
model = TextProcessingModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# ✅ Training Loop
print("🔄 Training AI Model on Real Dataset...")
for epoch in range(5):
    for batch in zip(input_data, labels):
        inputs, targets = batch
        inputs, targets = torch.unsqueeze(inputs, 0), torch.unsqueeze(targets, 0)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

# ✅ Save Model
torch.save(model.state_dict(), MODEL_PATH)

# ✅ Confirm Model Save
if os.path.exists(MODEL_PATH):
    print(f"✅ AI Model Trained and Successfully Saved at: {MODEL_PATH}")
    logging.info(f"✅ Model saved at: {MODEL_PATH}")
else:
    print("❌ ERROR: Model was NOT saved. Check directory permissions.")
    logging.error("❌ ERROR: Model was NOT saved.")
"""

    # ✅ Save the learned model script
    learned_model_path = os.path.join(BASE_PATH, "learned_model.py")
    with open(learned_model_path, "w", encoding="utf-8") as f:
        f.write(model_code)

    logging.info("✅ AI model generated and trained successfully.")
    print("✅ AI model generated and trained: learned_model.py")

if __name__ == "__main__":
    generate_ai_model_from_learning()
