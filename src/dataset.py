import torch
import cv2
import pandas as pd
import numpy as np
import config
import utils
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# def train_test_split(csv_path, split):
#     df_data = pd.read_csv(csv_path)
#     len_data = len(df_data)
#     # calculate the validation data sample length
#     valid_split = int(len_data * split)
#     # calculate the training data samples length
#     train_split = int(len_data - valid_split)
#     training_samples = df_data.iloc[:train_split][:]
#     valid_samples = df_data.iloc[-valid_split:][:]
#     return training_samples, valid_samples


class KeypointDataset(Dataset):
    def __init__(self, samples, path, augment=None):
        self.data = samples
        self.path = path
        self.size = config.IMG_SIZE
        self.augment = augment

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = cv2.imread(f"{self.path}/{self.data.iloc[index][0]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #orig_h, orig_w, channel = image.shape
        # resize the image into `resize` defined above
        #image = cv2.resize(image, (self.resize, self.resize))
        
        # get the keypoints
        keypoints = self.data.iloc[index][1:]
        keypoints = np.array(keypoints, dtype="uint8")
        
        # Incase any keypoint lie on the borders of the image, just moving it by one pixel.
        # This is done to prevent it from messing up the augmentation algorithms. :))
        num_zeros = 0
        for index, value in enumerate(keypoints):
            if value == self.size:
              keypoints[index] = self.size - 1
            if value == 0:
              num_zeros += 1
            if value < 0 or value > self.size:
              value = np.nan
        if num_zeros >= 2:
          index += 1
          return self.__getitem__(index)
        
        # reshape the keypoints
        keypoints = keypoints.reshape(8, 2)
        keypoints_initial = keypoints

        # rescale keypoints according to image resize
        #keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
        
        if self.augment is not None:
            transformed = self.augment(image=image, keypoints=keypoints)
            image = transformed["image"]
            keypoints = transformed["keypoints"]
        
        keypoints_augment = keypoints
        #Normalize the image
        image = image / 255.0

        # transpose for getting the channel size to index 0
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float)
        keypoint = torch.tensor(keypoints, dtype=torch.float)
        
        # Debug command -> To check when the tensor inputs are not of equal length
        if not torch.equal(torch.Tensor([8, 2]), torch.Tensor(list(keypoint.shape))):
          print(f"Keypoints Initial: {keypoints_initial}")
          print(f"Keypoints Augment: {keypoints_augment}")

        return {
            'image': image,
            'keypoints': keypoint,
        }

# get the training and validation data samples
# training_samples, valid_samples = train_test_split(f"{config.ROOT_PATH}/training_frames_keypoints.csv",
#                                                     config.TEST_SPLIT)


Transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.2),
    A.PiecewiseAffine(p=0.2),
    A.OneOf([
        A.Sharpen(),
        A.Emboss(),
    ], p=0.3),
    A.OneOf([
        A.HueSaturationValue(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
    ], p=0.4),
    A.OneOf([
        A.ImageCompression(p=0.3),
        A.ISONoise(p=0.4),
    ], p=0.6)
    #A.RandomFog(p=0.3),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)


# initialize the dataset - `FaceKeypointDataset()`
train_data = KeypointDataset(pd.read_csv(f"{config.ROOT_PATH}/Train.csv"), 
                                 config.ROOT_PATH, augment=Transform)
valid_data = KeypointDataset(pd.read_csv(f"{config.ROOT_PATH}/Validation.csv"), 
                                 config.ROOT_PATH)
# prepare data loaders
train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True)
valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)
print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

# for sample in train_data:
#     img = sample['image']
#     keypoint = sample['keypoints']
# whether to show dataset keypoint plots
if config.SHOW_DATASET_PLOT:
    utils.dataset_keypoints_plot(train_data)