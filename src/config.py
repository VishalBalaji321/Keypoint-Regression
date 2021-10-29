import torch

# constant paths
ROOT_PATH = 'Keypoint-Regression/input/ConeKeypoints'
OUTPUT_PATH = 'Keypoint-Regression/outputs'
DEFAULT_MODEL = 'resnet18'

IMG_SIZE = 80

# learning parameters
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
#TEST_SPLIT = 0.1
# show dataset keypoint plot
SHOW_DATASET_PLOT = False

print(f"PyTorch Detected DEVICE: {DEVICE}")