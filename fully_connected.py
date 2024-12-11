import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Class (modified to use only tabular features)
class CarDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Extract tabular features (drop non-required columns)
        tabular_features = torch.tensor(
            row.drop(['Price', 'Exterior Color', 'Interior Color', 'VIN', 'Dealer Name', 'Dealer Zip', 'Image Paths']).values.astype(np.float32)
        )

        # Label (Price)
        label = torch.tensor(row['Price'], dtype=torch.float32)

        return {'tabular': tabular_features, 'label': label}

# Load dataset
df = pd.read_csv('preprocessed_cars_data_with_onehot_more.csv')

# Remove non-tabular columns before training
tabular_columns = df.drop(['Price', 'Exterior Color', 'Interior Color', 'VIN', 'Dealer Name', 'Dealer Zip', 'Image Paths'], axis=1).columns

scaler = StandardScaler()
df['Price'] = scaler.fit_transform(df[['Price']])


# Dataset and DataLoader
dataset = CarDataset(df)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the Fully Connected Neural Network Model (for only tabular features)
class TabularModel(nn.Module):
    def __init__(self, num_tabular_features):
        super(TabularModel, self).__init__()
        
        self.tabular_net = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.Sigmoid(),
            nn.BatchNorm1d(128),  # BatchNorm layer
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),  # Another BatchNorm layer
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)  # Output: single value (price)
        )

    def forward(self, tabular):
        return self.tabular_net(tabular)


# Initialize the model with the correct number of tabular features
num_tabular_features = len(tabular_columns)  # Number of features in the tabular data
model = TabularModel(num_tabular_features=num_tabular_features).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3):

    train_losses = []
    val_losses = []
    epochs_run = 0

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            tabular_features = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(tabular_features)
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
                tabular_features = batch['tabular'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(tabular_features)
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
            torch.save(model.state_dict(), 'best_model.pth')
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
# Evaluate on test set
def evaluate_model(model, test_loader, scaler):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    absolute_deviations = []  # List to store absolute deviations for each prediction

    with torch.no_grad():
        for batch in test_loader:
            tabular_features = batch['tabular'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(tabular_features)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()

            # Save predictions and labels
            all_preds.append(outputs.squeeze().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Transform predictions and labels back to real price values
    all_preds = scaler.inverse_transform(np.concatenate(all_preds).reshape(-1, 1)).flatten()
    all_labels = scaler.inverse_transform(np.concatenate(all_labels).reshape(-1, 1)).flatten()

    # Calculate the absolute deviation for each sample
    absolute_deviations = np.abs(all_preds - all_labels)

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss (Standardized): {avg_test_loss:.4f}")

    # Calculate the average absolute deviation (mean absolute error)
    avg_absolute_deviation = np.mean(absolute_deviations)
    print(f"Average Absolute Deviation (Real Prices): {avg_absolute_deviation:.2f}")

    # Scatter plot of predicted vs actual prices
    plt.figure(figsize=(8, 8))
    plt.scatter(all_labels, all_preds, alpha=0.6, color='blue')
    plt.title('Predicted vs Actual Prices (Real Scale)')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], color='red', linestyle='--')  # Line of perfect prediction
    plt.show()

    # Histogram of predicted vs actual prices
    plt.figure(figsize=(8, 6))
    plt.hist([all_labels, all_preds], label=['Actual', 'Predicted'], bins=50, alpha=0.7)
    plt.title('Histogram of Actual and Predicted Prices (Real Scale)')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# Call the evaluation function
evaluate_model(model, test_loader, scaler)


