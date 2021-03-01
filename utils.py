from tqdm import tqdm
import torch
import config

def train_fn(model, data_loader, criterion, optimiser, scaler):
    model.train()
    final_loss = 0
    for data in data_loader:
        # get data
        inputs = data["image"]
        targets = data["mask"]
        bs, h, w = targets.size()
        
        # Move to device
        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)

        optimiser.zero_grad()

        with torch.cuda.amp.autocast():
            # forward pass throght model
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, targets)

        if scale != None:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        final_loss += loss.item()

    return outputs, final_loss/ len(data_loader)

def test_fn(model, data_loader, criterion):
    model.eval()
    final_loss = 0
    with torch.no_grad():
        for data in data_loader:
            # get data
            inputs = data["image"]
            targets = data["mask"]
            
            # Move to device
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            # forward pass throght model
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            final_loss += loss.item()

    return outputs, final_loss/ len(data_loader)
