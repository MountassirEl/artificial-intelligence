from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS,
    PATIENCE
)
from model import create_model
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
import torch
import matplotlib.pyplot as plt
import time
plt.style.use('ggplot')


# function for running training iterations
def train_one_epoch(train_data_loader, model, optimizer):
    
    # initialize tqdm progress bar
    # prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    # epoch loss averager
    epoch_loss_averager = Averager()
    
    for i, data in enumerate(train_data_loader):
        optimizer.zero_grad()
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        epoch_loss_averager.send(loss_value)
        losses.backward()
        optimizer.step()
    
        # update the loss value beside the progress bar for each iteration
        # prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        
    epoch_loss = epoch_loss_averager.value
    return epoch_loss

def train(train_loader, valid_loader, model, optimizer, num_epochs = NUM_EPOCHS):

    # initialize the model saver
    save_best_model = SaveBestModel()

    # early stopping trigger
    triggertimes = 0
    last_loss = save_best_model.best_valid_loss
    
    # train and validation loss lists
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(num_epochs):
        
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train_one_epoch(train_loader, model, optimizer)
        val_loss = validate(valid_loader, model)
        end = time.time()
        period = ((end - start) / 60)
        print(f"Epoch :{epoch+1} ({period:.3f} min) train loss: {train_loss:.3f} validation loss: {val_loss:.3f} EarlyStop trigger {triggertimes}")
        
        # store losses
        train_loss_list.append(train_loss)
        valid_loss_list.append(val_loss)

        # check for early stopping
        if val_loss > last_loss:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print('Early stopping!')
                return model
        else:
            trigger_times = 0
        
        # update last loss
        last_loss = val_loss
        
        # save the best model till now if we have
        # the least loss in the current epoch
        save_best_model(val_loss, epoch, model, optimizer)
        
        # save the current epoch model
        save_model(epoch, model, optimizer)
        
    # save loss plot
    save_loss_plot(OUT_DIR, train_loss_list, valid_loss_list)
    return model


# function for running validation iterations
def validate(valid_data_loader, model):
    
    # initialize tqdm progress bar
    # prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    # epoch loss averager
    epoch_loss_averager = Averager()
    
    for i, data in enumerate(valid_data_loader):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        epoch_loss_averager.send(loss_value)
        
        # update the loss value beside the progress bar for each iteration
        # prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    epoch_loss = epoch_loss_averager.value
    return epoch_loss

