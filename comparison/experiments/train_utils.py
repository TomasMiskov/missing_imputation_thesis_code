import torch
from torch import nn
from torcheval.metrics import BinaryAUROC, BinaryF1Score, MeanSquaredError, R2Score
import random
import numpy as np

def train1epoch(model, optimizer, criterion, train_loader, val_loader, device):
    train_loss = 0.0
    model.train()
    
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        outputs = model(data).to(device)
        
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Calculate the average loss over an epoch
    train_loss /= len(train_loader)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            outputs = model(data).to(device)
            loss = criterion(outputs, label)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    return val_loss

# ====================================================

def train(model, train_loader, val_loader, epochs, patience, regression_flag, device, seed, verbose = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.to(device)
    if regression_flag:
        criterion = nn.MSELoss()
    else:
        # weight the classes to handle imbalanced datasets
        _, counts = np.unique(train_loader.dataset.tensors[1], return_counts=True)
        class_weights = torch.tensor((1.0 / counts) * len(train_loader.dataset.tensors[1]) / 2.0)
        pos_weight = torch.tensor([class_weights[1]]).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  
    
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        val_loss = train1epoch(model, optimizer, criterion, train_loader, val_loader, device)

        if verbose and (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}, val loss: {val_loss}")
              
        # Handle early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                model.load_state_dict(best_model)
                break

# ====================================================

def getPredictions(model, test_loader, device):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, y in test_loader:
            data = data.to(device)
            outputs = model(data).to(device)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(y.cpu().numpy())
    return np.concatenate(y_pred), np.concatenate(y_true)