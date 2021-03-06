from tqdm import tqdm
import glob
import torch
import torch.nn as nn
import numpy as np
import cv2
import os

from sklearn import model_selection

import config
import dataset
import utils
import unet_models


def dice_loss(pred, target):
    smooth = 1.
    intersection = (pred*target).sum(axis=[0,2,3])

    return 1 - ((2.*intersection + smooth) /
                (pred.sum(axis=[0,2,3]) + target.sum(axis=[0,2,3]) + smooth)).mean()

def pred_to_hooman(pred):
    pred = torch.argmax(pred, 0).type(torch.uint8)
    pred = pred.detach().cpu().numpy()
    ret = np.zeros((pred.shape[0], pred.shape[1], 3))
    for class_num, colour in enumerate([[128,255,0], [0,255,255], [255,0,127],
                                        [255,0,255], [255,0,0]]):
        ret[pred==class_num] = colour

    return ret

def run_training():
    mask_paths = glob.glob(f'{config.DATA_DIR}/cat_masks/*.png')
    image_paths = [f'{config.DATA_DIR}/imgs/{x.split("/")[-1]}' for x in mask_paths]

    (
        train_im,
        test_im,
        train_mask,
        test_mask
    ) = model_selection.train_test_split(
        image_paths, mask_paths, test_size=0.1, random_state=69
    )
   
    train_dataset = dataset.MaskImDataset(
       image_paths = train_im,
       mask_paths = train_mask,
       resize = config.RESIZE
    )

    test_dataset = dataset.MaskImDataset(
        image_paths = test_im,
        mask_paths = test_mask,
        resize = config.RESIZE
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    # Load model
    model = unet_models.ResnetUnet(class_num=5)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)
    if config.LOAD != None:
        checkpoint = torch.load(config.LOAD)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
    else:
        best_loss = 9999
        start_epoch = 0

    model.to(config.DEVICE)

    if config.fp16:
        # Scaler for fp16 traing (mixed precision)
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.3, patience=5, verbose=True
    )

    for epoch in tqdm(range(start_epoch, config.EPOCHS)):
        _, train_loss = utils.train_fn(model, train_loader, criterion, optimiser, scaler)
        prediction, test_loss = utils.test_fn(model, test_loader, criterion)
        print(f'\rEpoch {epoch} Train Loss={train_loss} Test loss={test_loss}')
        
        scheduler.step(test_loss)
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss': test_loss,
                        }, f'./models/unet_{epoch}.pt')
            print('Model Saved')

        if not config.HEADLESS:
            look_at_training = pred_to_hooman(prediction[0,:,:,:])
            cv2.imshow('Prediction', look_at_training)
            cv2.waitKey(30)

if __name__ == "__main__":
    # Test if models dir exsits make if not
    if not os.path.exists('./models'):
        os.mkdir('./models')
    run_training()
    cv2.destroyAllWindows()
