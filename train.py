import torch
import numpy as np
import random
import os
from torch.utils import data
from sklearn.model_selection import KFold
from data_loaders import Dataset
from data_augment import DataAugment
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
import io
import PIL.Image
import sys
import copy
from torchvision.transforms import ToTensor

from collections import OrderedDict
from utils import expand_as_one_hot

import matplotlib.pyplot as plt
plt.style.use("ggplot")

#Specify Which Model and Loss to Import
from models import UNet3D
from models_modified import Modified3DUNet
from losses import GeneralizedDiceLoss
import shutil
from sys import argv

def gen_subplot(scans_orig, scans_aug, mask, prediction, epoch_nr, iteration, aug_used):
    _, indices = prediction.max(0)
    indices = indices.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    img_size = indices.shape[2]
    slices = [int(img_size/4), int(img_size/4*2), int(img_size/4*3)] # Slices to display
    plt.figure()
    for row in range(1,4):
        plt.subplot(3, 8, 1 + (row - 1) * 8)
        # Showing the t1ce scan
        plt.imshow(scans_orig[1, :, :, slices[row - 1]], cmap='gray')
        plt.ylabel("Slice {}".format(slices[row - 1]))
        if row == 1:
            plt.title('Orig')
        plt.subplot(3, 8, 2 + (row-1)*8)
        #Showing the t1ce scan
        plt.imshow(scans_aug[1, :, :, slices[row-1]], cmap='gray')
        plt.ylabel("Slice {}".format(slices[row-1]))
        if row==1:
            plt.title('Aug: ' + aug_used)
        plt.subplot(3, 8, 3 + (row-1)*8)
        plt.imshow(mask[:, :, slices[row-1]])
        if row==1:
            plt.title('Mask')
        plt.subplot(3, 8, 4 + (row-1)*8)
        plt.imshow(prediction[0,:,:,slices[row-1]])
        if row==1:
            plt.title('Cl 0')

        plt.subplot(3, 8, 5 + (row - 1) * 8)
        plt.imshow(prediction[1, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 1')

        plt.subplot(3, 8, 6 + (row - 1) * 8)
        plt.imshow(prediction[2, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 2')

        plt.subplot(3, 8, 7 + (row - 1) * 8)
        plt.imshow(prediction[3, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 3')

        plt.subplot(3, 8, 8 + (row - 1) * 8)
        plt.imshow(indices[:, :, slices[row-1]])
        if row == 1:
            plt.title('Pred')
    plt.suptitle("Epoch {} Iteration {}".format(str(epoch_nr), str(iteration)))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image

########################################################################################################################
# Code for training a 3D-Unet, models and losses must be specified. Currently uses 3D-UNet and Generalized dice loss
# obtained form wolny/pytorch-3dunet

# To specify:

# Paths where to load data from and save the models to
#preprocessed_data_path = r"TrainingPreprocessed"
#save_model_path = r"Data_Aug_Experiments"

preprocessed_data_path = argv[1]
save_model_path = argv[2]
run_name = argv[3]
save_losses = argv[4]

# Create the folder to store preprocessed data in, exit if folder already exists.
if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)
else:
    print("Folder to store data experiment already exists")    
    shutil.rmtree(save_model_path)
    print("Folder removed")    
    os.mkdir(save_model_path)
    print("Folder created")  
    #sys.exit()

if not os.path.isdir(save_losses):
    os.mkdir(save_losses)
else:
    print("Folder to store data experiment already exists")    
    shutil.rmtree(save_losses)
    print("Folder removed")    
    os.mkdir(save_losses)
    print("Folder created")  
    #sys.exit()

# Specify which data augmentations to use on the fly (each applied with 50% probability). Possible values:
# ['Elastic', 'Flip', 'Rotate','Gamma','Scale', 'Noise']. Create empty array if none wanted.
augmentations_to_use = ['Flip', 'Rotate', 'Gamma', 'Scale', 'Noise']

# Training Parameters
batch_size = 2
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 4}
max_epochs = 10 # 300

# Model Parameters
in_channels = 4
n_classes = 4
base_n_filter = 16
n_folds = 5  # Number of folds in cross-validation

#patience = 30

# initialize the early_stopping object
#early_stopping = EarlyStopping(patience=patience, verbose=True)

##############################################################################

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/' + run_name)

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Get paths and names (IDS) of folders that store the multimodal training data
folder_paths = []
folder_ids = []
for subdir in os.listdir(preprocessed_data_path):
    folder_paths.append(os.path.join(preprocessed_data_path, subdir))
    folder_ids.append(subdir)

# Shuffle them around, keeping same seed to make sure same shuffling is used if training is interrupted and needs to be continued
random.seed(4)
random.shuffle(folder_paths)
random.seed(4)
random.shuffle(folder_ids)

# Setup KFold Cross Validation
kf = KFold(n_splits=n_folds, shuffle=False)  # Shuffle=false to get the same shuffling scheme every run
fold_nr = 1

# Training Loop
for fold in kf.split(folder_paths):
    iter_nr = 1
    train_idx = fold[0]
    valid_idx = fold[1]
    train_set = Dataset([folder_paths[i] for i in train_idx], [folder_ids[i] for i in train_idx])
    valid_set = Dataset([folder_paths[i] for i in valid_idx], [folder_ids[i] for i in valid_idx])
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # Model
    model = UNet3D(in_channels, n_classes, False, base_n_filter, 'crg', 8)
    #model = Modified3DUNet(in_channels, n_classes, base_n_filter)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    #If training was interrupted (need to change epoch loop range as well):
    #checkpoint = torch.load("/home/ajurgens/Brats2019/Model_Saves_V4/Fold_1_Epoch_140.tar")
    #model.load_state_dict(checkpoint['model_state_dict'])

    # Loss and optimizer
    criterion = GeneralizedDiceLoss(1e-5, None, None, False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=10**-7)
    model.to(device)

    #If training was interrupted (need to change epoch loop range as well):
    #model.train()

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        train_losses = []
        for batch, labels in train_loader:
            batch_orig = copy.deepcopy(batch)  # Save batch before augmentation for plotting outputs

            # Data Augment if augmentations were given
            if not len(augmentations_to_use) == 0:
                augmenter = DataAugment(batch, labels, augmentations_to_use, True)
                batch, labels, augmentation_parameters = augmenter.augment()

            # If scaling to a bigger volume was performed, randomly sample a 128x128x128 patch again or it wont fit into GPU memory
            if batch.shape[2] > 128:
                x_orig = random.sample(range(batch.shape[2] - 127), 1)[0]
                y_orig = random.sample(range(batch.shape[3] - 127), 1)[0]
                z_orig = random.sample(range(batch.shape[4] - 127), 1)[0]
                batch = batch[:, :, x_orig: x_orig + 128, y_orig: y_orig + 128, z_orig: z_orig + 128]
                labels = labels[:, x_orig: x_orig + 128, y_orig: y_orig + 128, z_orig: z_orig + 128]

            # Transfer batch and labels to GPU
            scans, masks = batch.to(device), labels.to(device)
            # Run through network
            output = model(scans)

            # Save images of network output every 100 iterations
            if (iter_nr % 100 == 0):
                subplot_img = gen_subplot(batch_orig[0], batch[0], labels[0], output[0], epoch, iter_nr, augmentation_parameters)
                writer.add_image('{}'.format(run_name), subplot_img, iter_nr)

            masks = expand_as_one_hot(masks, n_classes)
            train_loss = criterion(output, masks)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            # Log training loss to tensorboard
            writer.add_scalar('Train Loss', train_loss.item(), iter_nr)
            iter_nr += 1

        # Get training loss after every epoch
        train_loss_ep = np.mean(train_losses)
        writer.add_scalar('TrainPE Fold {}'.format(fold_nr), train_loss_ep, epoch)
        valid_losses = []
        with torch.no_grad():
            for batch, labels in valid_loader:
                scans, masks = batch.to(device), labels.to(device)
                output = model(scans)
                masks = expand_as_one_hot(masks, n_classes)
                valid_loss = criterion(output, masks)
                valid_losses.append(valid_loss.item())
        valid_loss_ep = np.mean(valid_losses)
        # Log valid loss to tensorboard
        writer.add_scalar('Valid Loss per Epoch', valid_loss_ep, epoch)
        elapsed_time = time.time() - start_time

        print('Fold [{}/{}], Epoch [{}/{}], Train Loss {:.10f}, Valid Loss {:.10f}, Time_{}'.format(fold_nr, n_folds, epoch, max_epochs, train_loss_ep, valid_loss_ep, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        losses = open("{}/Losses/{}.txt".format(save_model_path, run_name), "a")
        losses.write('Fold [{}/{}], Epoch [{}/{}], Train Loss {:.10f}, Valid Loss {:.10f}, Time {}\n'.format(fold, n_folds,epoch, max_epochs,train_loss_ep,valid_loss_ep,time.strftime("%H:%M:%S",time.gmtime(elapsed_time))))
        losses.close()

        # Save the model parameters
        if (epoch % 10 == 0):
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, "{}/Fold_{}_Epoch_{}.tar".format(save_model_path, fold_nr, epoch))
    fold_nr = fold_nr + 1