import albumentations
import torch
import numpy as np
import cv2


model_path = './model/Unet.pt'
vid_path = './videos/vid.mp4'
vid_out = './videos/out_vid.mp4'

DEVICE = 'cuda'
# Load model set in eval mode
model = torch.load(model_path)
model.to(DEVICE)
model.eval()

# Augmentation (normilisation) init
aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])

# cv2 captuer vid
cap = cv2.VideoCapture(vid_path)

def pred_to_hooman(pred):
    pred = torch.squeeze(pred, 0)
    pred = torch.argmax(pred, 0).type(torch.uint8)
    pred = pred.detach().cpu().numpy()
    ret = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_num, colour in enumerate([[128,255,0], [0,255,255], [255,0,127],
                                        [255,0,255], [255,0,0]]):
        ret[pred==class_num] = colour

    return ret

ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (256,256))
        show_frame = np.copy(frame)
        augmented = aug(image=frame)
        frame = augmented["image"]
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.tensor(frame, dtype=torch.float)
        frame = torch.unsqueeze(frame, 0)
        frame = frame.to(DEVICE)
        
        prediction = model(frame)
        prediction = pred_to_hooman(prediction)
        cat = np.concatenate((show_frame, prediction), axis=1)
        cat = cv2.resize(cat, (1280,720))
        cv2.imshow('cat', cat)
        cv2.imshow('vid', show_frame)
        cv2.imshow('pred', prediction)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
    
cv2.destroyAllWindows()
cap.release()

