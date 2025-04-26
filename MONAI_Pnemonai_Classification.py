# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Install MONAI and Gradio
!pip install -q monai[all] gradio

# Step 3: Imports
import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from monai.networks.nets import densenet121
from sklearn.metrics import classification_report, confusion_matrix
import gradio as gr

# Step 4: Paths
data_dir = "/content/drive/My Drive/MONAI_Pneumonai_Classification/dataset"
model_path = "/content/drive/My Drive/MONAI_Pneumonai_Classification/pneumonia_densenet121.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 5: Dataset
transform = transforms.Compose([
    transforms.Grayscale(),              # Ensure 1 channel
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Step 6: Model Setup
model = densenet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Step 7: Training
epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# Step 8: Save Model
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")

# Step 9: Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Results
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

report = classification_report(y_true, y_pred, target_names=train_dataset.classes)
print("Classification Report:\n", report)

# Step 10: Gradio UI for Prediction
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def predict_image(img):
    import PIL
    img = img.convert("L").resize((224, 224))  # grayscale
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = prob.argmax().item()
    return {
        train_dataset.classes[0]: float(prob[0]),
        train_dataset.classes[1]: float(prob[1])
    }

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Pneumonia Detection from Chest X-ray"
).launch()