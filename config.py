# encoding: utf-8
"""
Global configuration file

Centralizes dataset paths, result save directories, training hyperparameters,
loss weights, and noise layer settings. main.py reads directly from this file,
making it easy to maintain and reuse when sharing the code.
"""

# =========================== Paths ===========================
# Dataset root directory (should contain train/ and val/ subdirectories)
DATA_ROOT = r"D:\Python_Dataset\COCO"

# Base directory for saving results (checkPoints / trainPics / validationPics / logs / codes / testPics)
SAVE_ROOT = r"./training/"

# Pretrained checkpoint path (loaded when SAVED_MODE is True)
SAVED_MODE_PATH = r"D:\Python_code\zjz_work6\UnoiEncoder\training\testforCOCO_fewer_2025-11-24-13_41_53\checkPoints\checkpoint_30.pth"

# Resume training from the pretrained checkpoint
SAVED_MODE = False


# ========================= Training hyperparameters =========================
DATASET = "train"          # train | val | test
WORKERS = 0                # number of data loading workers
BATCH_SIZE = 16            # input batch size
IMAGE_SIZE = 128           # input image size
MESSAGE_LENGTH = 64        # secret message bit length
NITER = 100                # number of training epochs
LR = 0.001                 # learning rate
DECAY_ROUND = 10           # decay learning rate by 0.5 every DECAY_ROUND epochs
BETA1 = 0.5                # beta1 for Adam
CUDA = True                # whether to enable cuda
NGPU = 1                   # number of GPUs to use
BS_SECRET = 32             # batch size for the secret message
BETA = 0.75                # beta hyperparameter
REMARK = ""                # experiment remark
TEST = ""                  # test mode: path to the test image directory
HOSTNAME = "HFSN"          # running server hostname (used as the experiment directory prefix)
DEBUG = False              # debug mode does not create folders
LOG_FREQUENCY = 10         # frequency of printing logs to the console
RESULT_PIC_FREQUENCY = 500  # frequency of saving result images


# ========================= Loss weights =========================
LPIPS_WEIGHT = 0.3
SSIM_WEIGHT = 0.5
MSE_WEIGHT = 1
DECODE_WEIGHT = 3


# ========================= Noise layer settings =========================
NOISE_LAYERS = [
    'Resize',
    'GaussianBlur',
    'Dropout',
    'Cropout',
    'MedianBlur',
    'GaussianNoise',
    'SaltAndPepper',
    'JPEG',
    'GridCrop',
    'Identity',
]
