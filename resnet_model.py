import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Class
class CarDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Extract tabular features (drop non-required columns)
        tabular_features = torch.tensor(
            row.drop(['Image Paths', 'Price', 'Exterior Color', 'Interior Color', 'VIN', 'Dealer Name', 'Dealer Zip']).values.astype(np.float32)
        )
        
        # Process image
        image_paths = eval(row['Image Paths'])  # Parse the stringified list
        images = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        # Select the first image if available, otherwise use a placeholder
        if images:
            image = images[0]
        else:
            image = torch.zeros((3, 224, 224))  # Placeholder for missing images
        
        # Label (Price)
        label = torch.tensor(row['Price'], dtype=torch.float32)

        return {'image': image, 'tabular': tabular_features, 'label': label}

# Transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
df = pd.read_csv('preprocessed_cars_data_with_onehot_more.csv')
# Initialize scaler and fit on 'Price' column
scaler = StandardScaler()
df['Price'] = scaler.fit_transform(df[['Price']])  # Standardize the Price column

dataset = CarDataset(df, transform=transform)

# Split dataset into train, validation, and test sets (80-10-10 split)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Define the Multi-Modal Neural Network
class MultiModalModel(nn.Module):
    def __init__(self, num_tabular_features):
        super(MultiModalModel, self).__init__()
        
        # Pre-trained ResNet model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)  # Custom output layer
        
        # Tabular feature processor
        self.tabular_net = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # BatchNorm layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # Another BatchNorm layer
        )
        
        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: single value (price)
        )

    def forward(self, image, tabular):
        # Extract features from the image
        image_features = self.resnet(image)
        
        # Process tabular features
        tabular_features = self.tabular_net(tabular)
        
        # Concatenate both feature vectors
        combined_features = torch.cat((image_features, tabular_features), dim=1)
        
        # Pass through combined network
        output = self.combined_net(combined_features)
        return output

# Determine the correct number of tabular features
tabular_columns = df.drop(['Image Paths', 'Price', 'Exterior Color', 'Interior Color', 'VIN', 'Dealer Name', 'Dealer Zip'], axis=1).columns
num_tabular_features = len(tabular_columns)  # Number of features in the tabular data

# Initialize the model with the correct number of tabular features
model = MultiModalModel(num_tabular_features=num_tabular_features).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=2):

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            tabular_features = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images, tabular_features)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                tabular_features = batch['tabular'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, tabular_features)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        # Store losses for plotting
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model_mmnn.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

        epochs_run += 1

    # Plotting training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs_run+1), train_losses[:epochs_run], label='Training Loss', color='blue')
    plt.plot(range(1, epochs_run+1), val_losses[:epochs_run], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Evaluate on test set
def evaluate_model(model, test_loader, scaler):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    absolute_deviations = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            tabular_features = batch['tabular'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(images, tabular_features)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()

            # Save predictions and labels
            all_preds.append(outputs.squeeze().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Transform predictions and labels back to real price values
    all_preds = scaler.inverse_transform(np.concatenate(all_preds).reshape(-1, 1)).flatten()
    all_labels = scaler.inverse_transform(np.concatenate(all_labels).reshape(-1, 1)).flatten()

    # Calculate the average absolute deviation
    absolute_deviations = np.abs(all_preds - all_labels)
    avg_absolute_deviation = np.mean(absolute_deviations)
    print(f"Average Absolute Deviation: {avg_absolute_deviation:.2f}")

    # Scatter plot of predicted vs actual prices
    plt.figure(figsize=(8, 8))
    plt.scatter(all_labels, all_preds, alpha=0.6, color='blue')
    plt.title('Predicted vs Actual Prices (Real Scale)')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], color='red', linestyle='--')
    plt.show()



evaluate_model(model, test_loader, scaler)

def visualize_sample_images(model, test_loader):
    model.eval()
    
    sample_images = []
    sample_preds = []
    sample_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, batch['tabular'].to(device))
            
            sample_images.extend(images.cpu().numpy())  # Store the images
            sample_preds.extend(outputs.cpu().numpy())  # Store the predictions
            sample_labels.extend(labels.cpu().numpy())  # Store the actual labels
            
            # Break after one batch for quick visualization
            break
    
    # Convert to numpy arrays for easier indexing
    sample_images = np.array(sample_images)
    sample_preds = np.array(sample_preds).flatten()  # Flatten to make sure it's a 1D array
    sample_labels = np.array(sample_labels).flatten()  # Flatten to make sure it's a 1D array
    
    # Visualize the first 5 samples
    for i in range(5):
        img = sample_images[i].transpose(1, 2, 0)  # Convert from CHW to HWC format
        img = np.clip(img, 0, 1)  # Ensure pixel values are in [0, 1]
        
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pred: {sample_preds[i]:.2f}\nActual: {sample_labels[i]:.2f}")
    
    plt.show()

# Visualize sample images with predictions
visualize_sample_images(model, test_loader)

