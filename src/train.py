import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
import utils
import timm

from model import KeypointResNet, KeypointEfficientNet, KeypointCustom
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm

matplotlib.style.use('ggplot')

# training function
def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)

    #Creating a gradScaler at the beginning of the training
    scaler = GradScaler()

    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        with autocast():
            outputs = model(image)
            loss = criterion(outputs, keypoints)
        
        train_running_loss += loss.item()
        scaler.scale(loss).backward()
        #loss.backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        
    train_loss = train_running_loss/counter
    return train_loss

# validatioon function
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if (epoch+1) % 1 == 0 and i == 0:
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

# model 
#model = KeypointResNet(pretrained=True, requires_grad=True, model_name=config.RESNET_MODEL).to(config.DEVICE)
#model = KeypointEfficientNet(pretrained=True, requires_grad=True)
model = KeypointCustom(isPretrained=True, requires_grad=True)
model = model.return_loaded_model().to(config.DEVICE)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LR)

# we need a loss function which is good for regression like SmmothL1Loss ...
# ... or MSELoss -> Use this when working with GrayScale Images
criterion = nn.SmoothL1Loss()

train_loss = []
val_loss = []

for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    train_epoch_loss = fit(model, train_loader, train_data)
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')
    if (epoch % 5 == 0):
        torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/model_{epoch}.pth")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.show()
torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/model_final.pth")
print('DONE TRAINING')