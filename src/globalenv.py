import sys
from pathlib import Path

SRC_PATH = Path(__file__).absolute().parent
ROOT_PATH = SRC_PATH.parent
sys.path.append(str(SRC_PATH))

LOGGER_BUFFER_LOCK = False
SPLIT = "————————————————————————————————————————————————————"

GLOBAL_SEED = 233
TEST_RESULT_DIRNAME = "test_result"
TRAIN_LOG_DIRNAME = "log"
CONFIG_DIR = "config"
CONFIG_FILEPATH = "config/config.yaml"
LMDB_DIRPATH = ROOT_PATH / "lmdb"
METRICS_LOG_DIRPATH = ROOT_PATH / "metrics_log"
OPT_FILENAME = "CONFIG.yaml"
LOG_FILENAME = "run.log"
LOG_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
INPUT = "input"
OUTPUT = "output"
GT = "GT"
STRING_FALSE = "False"
SKIP_FLAG = "q"
DEFAULTS = "defaults"
HYDRA = "hydra"

INPUT_FPATH = "input_fpath"
GT_FPATH = "gt_fpath"

DEBUG = "debug"
BACKEND = "backend"
CHECKPOINT_PATH = "checkpoint_path"
LOG_DIRPATH = "log_dirpath"
IMG_DIRPATH = "img_dirpath"
DATALOADER_N = "dataloader_num_worker"
VAL_DEBUG_STEP_NUMS = "val_debug_step_nums"
VALID_EVERY = "valid_every"
LOG_EVERY = "log_every"
AUGMENTATION = "aug"
RUNTIME_PRECISION = "runtime_precision"
NUM_EPOCH = "num_epoch"
NAME = "name"
LOSS = "loss"
TRAIN_DATA = "train_ds"
VALID_DATA = "valid_ds"
TEST_DATA = "test_ds"
GPU = "gpu"
RUNTIME = "runtime"
CLASS = "class"
MODELNAME = "modelname"
BATCHSIZE = "batchsize"
VALID_BATCHSIZE = "valid_batchsize"
LR = "lr"
CHECKPOINT_MONITOR = "checkpoint_monitor"
MONITOR_MODE = "monitor_mode"
COMMENT = "comment"
EARLY_STOP = "early_stop"
AMP_BACKEND = "amp_backend"
AMP_LEVEL = "amp_level"
VALID_RATIO = "valid_ratio"

LTV_LOSS = "ltv"
COS_LOSS = "cos"
SSIM_LOSS = "ssim_loss"
L1_LOSS = "l1_loss"
COLOR_LOSS = "l_color"
SPATIAL_LOSS = "l_spa"
EXPOSURE_LOSS = "l_exp"
WEIGHTED_LOSS = "weighted_loss"
PSNR_LOSS = "psnr_loss"
HIST_LOSS = "hist_loss"
INTER_HIST_LOSS = "inter_hist_loss"
VGG_LOSS = "vgg_loss"
SPARSE_WEIGHT_LOSS = "sparse_weight_loss"
REG_SMOOTH_LOSS = "reg_smooth_loss"
REG_MONO_LOSS = "reg_mono_loss"
BRIGHTEN_LOSS = "brighten_loss"
DARKEN_LOSS = "darken_loss"

PSNR = "psnr"
SSIM = "ssim"

VERTICAL_FLIP = "v-flip"
HORIZON_FLIP = "h-flip"
DOWNSAMPLE = "downsample"
RESIZE_DIVISIBLE_N = "resize_divisible_n"
CROP = "crop"
LIGHTNESS_ADJUST = "lightness_adjust"
CONTRAST_ADJUST = "contrast_adjust"

BUNET = "bilateral_upsample_net"
UNET = "unet"
HIST_UNET = "hist_unet"
PREDICT_ILLUMINATION = "predict_illumination"
FILTERS = "filters"

MODE = "mode"
COLOR_SPACE = "color_space"
BETA1 = "beta1"
BETA2 = "beta2"
LAMBDA_SMOOTH = "lambda_smooth"
LAMBDA_MONOTONICITY = "lambda_monotonicity"
MSE = "mse"
L2_LOSS = "l2_loss"
TV_CONS = "tv_cons"
MN_CONS = "mv_cons"
WEIGHTS_NORM = "wnorm"
TEST_PTH = "test_pth"

LUMA_BINS = "luma_bins"
CHANNEL_MULTIPLIER = "channel_multiplier"
SPATIAL_BIN = "spatial_bin"
BATCH_NORM = "batch_norm"
NET_INPUT_SIZE = "net_input_size"
LOW_RESOLUTION = "low_resolution"
ONNX_EXPORTING_MODE = "onnx_exporting_mode"
SELF_SUPERVISED = "self_supervised"
COEFFS_TYPE = "coeffs_type"
ILLU_MAP_POWER = "illu_map_power"
GAMMA = "gamma"
MATRIX = "matrix"
GUIDEMAP = "guidemap"
USE_HSV = "use_hsv"

USE_WAVELET = "use_wavelet"
NON_LOCAL = "use_non_local"
USE_ATTN_MAP = "use_attn_map"
ILLUMAP_CHANNEL = "illumap_channel"
HOW_TO_FUSE = "how_to_fuse"
SHARE_WEIGHTS = "share_weights"
BACKBONE = "backbone"
ARCH = "arch"
N_BINS = "n_bins"
BACKBONE_OUT_ILLU = "backbone_out_illu"
ADAINT_FIX_INIT = "adaint_fix_init"
CONV_TYPE = "conv_type"
HIST_AS_GUIDE_ = "hist_as_guide"
ENCODER_USE_HIST = "encoder_use_hist"
GUIDE_FEATURE_FROM_HIST = "guide_feature_from_hist"
NC = "channel_nums"

ILLU_MAP = "illu_map"
INVERSE_ILLU_MAP = "inverse_illu_map"
BRIGHTEN_INPUT = "brighten_input"
DARKEN_INPUT = "darken_input"


TRAIN = "train"
TEST = "test"
VALID = "valid"
ONNX = "onnx"
CONDOR = "condor"
IMAGES = "images"

DEFORM = "deform"
NORMAL = "normal"
NORMAL_EX_LOSS = "normal_ex_loss"
INVERSE = "inverse"
BRIGHTEN_OFFSET = "brightness_offset"
DARKEN_OFFSET = "darken_offset"