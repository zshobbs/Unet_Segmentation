import albumentations
import torch
import numpy as np
import cv2


class MaskImDataset:
    def __init__(self, image_paths, mask_paths, resize=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.resize = resize
        self.aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        mask = cv2.imread(self.mask_paths[item], cv2.IMREAD_GRAYSCALE)

        if self.resize is not None:
            image = cv2.resize(image, (self.resize, self.resize))
            mask = cv2.resize(mask, (self.resize, self.resize))

        # add augmentation to image
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1))

        # Convert mask to onehot
        mask = torch.from_numpy(mask).type(torch.long)
        mask.unsqueeze_(0)
        mask_onehot = torch.LongTensor(5, mask.size(1), mask.size(2))
        mask_onehot.zero_()
        mask_onehot.scatter_(0, mask.data, 1)

        return {"image": torch.tensor(image, dtype=torch.float),
                "mask": mask_onehot}

if __name__ == "__main__":
    # Fast test see if working as exspected
    import glob

    mp = glob.glob('./comma10k/cat_masks/*.png')
    ip = [f'./comma10k/imgs/{x.split("/")[-1]}' for x in mp]
    t = MaskImDataset(ip, mp)
    print(t[0])
