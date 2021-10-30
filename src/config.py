import torch

# constant paths
ROOT_PATH = 'Keypoint-Regression/input/ConeKeypoints'
OUTPUT_PATH = 'Keypoint-Regression/outputs'

CURRENT_MODEL = 'tf_mobilenetv3_small_100'

IMG_SIZE = 80

# learning parameters
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50
DECAY_EPOCH = 100
DECAY_LR = 0.0005

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# show dataset keypoint plot
SHOW_DATASET_PLOT = False

print(f"PyTorch Detected DEVICE: {DEVICE}")

# Models
models_to_evaluate = [   
    'efficientnet_b0',
    'efficientnet_b1_pruned',
    'efficientnet_b2_pruned',

    'tf_efficientnet_lite0',
    'tf_efficientnet_lite1',
    'tf_efficientnet_lite2',
    'tf_efficientnet_lite3',

    'efficientnetv2_rw_m',
    'efficientnetv2_rw_s',
    'tf_efficientnetv2_s_in21ft1k',
    'tf_efficientnetv2_m_in21ft1k',

    'inception_v3',
    'inception_v4',

    'mixnet_l',
    'mixnet_m',
    'mixnet_s',

    'mobilenetv3_large_100_miil_in21k',
    'mobilenetv3_rw',
    'tf_mobilenetv3_small_075',
    'tf_mobilenetv3_small_100',
    'mobilenetv2_110d',
    'mobilenetv2_120d',

    'resnet18d',
    'resnet26d',
    'resnet34d',
    'resnet50d'
]