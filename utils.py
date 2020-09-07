from tqdm import tqdm
import torch
import config

def train_fn(model, data_loader, criterion, optimiser):
    model.train()
    final_loss = 0
    print('Training')
    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        # get data
        inputs = data["image"]
        targets = data["mask"]
        bs, c, h, w = targets.size()
        
        # Move to device
        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)

        optimiser.zero_grad()

        # forward pass throght model
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)
        optimiser.step()
        final_loss += loss.item()

    tk.close()
    return outputs, final_loss/ len(data_loader)

def test_fn(model, data_loader, criterion):
    model.eval()
    final_loss = 0
    print('Test')
    tk = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for data in tk:
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

    tk.close()
    return outputs, final_loss/ len(data_loader)
