import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_bce
# TODO: 修改TimeSeriesForecastingDataset和SimpleTimeSeriesForecastingRunner
from basicts.data import StockPricePredictionDataset
from basicts.runners import SimpleStockPricePredictionRunner
from basicts.archs import STID_S


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "T-SIDE model configuration"
CFG.RUNNER = SimpleStockPricePredictionRunner
CFG.DATASET_CLS = StockPricePredictionDataset
CFG.DATASET_NAME = "STOCK-IND"
CFG.DATASET_TYPE = "Stock Industry Data"
CFG.DATASET_INPUT_LEN = 5 # 时间窗口
CFG.DATASET_OUTPUT_LEN = 5 # 时间窗口
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0
CFG.RESCALE = False

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STID"
CFG.MODEL.ARCH = STID_S

CFG.MODEL.PARAM = {
    "input_len": 5,
    "input_dim": 52, # feature dim: 开盘 收盘 最高 最低 成交量 换手率 DOW DOM DOQ DOY 42板块列 具体用哪些列在simple_spp_runner.py定义
    "embed_dim": 32,
    "output_len": 1, 
    "num_layer": 3,
    "if_node": True,
    "num_nodes": 1,
    "node_dim": 32,
    "if_D_i_W": True,
    "temp_dim_diw": 32,
    "day_of_week_size": 7, # TODO
    "if_D_i_M": True,
    "temp_dim_dim": 32,
    "day_of_month_size": 31, # 1-31
    "if_D_i_Q": True,
    "temp_dim_diq": 32,
    "day_of_quarter_size": 92, # 1-92
    "if_D_i_Y": True,
    "temp_dim_diy": 32,
    "day_of_year_size": 365, # 2-366
}
#CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]  # traffic flow, time in day
#CFG.MODEL.TARGET_FEATURES = [0] # traffic flow

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_bce
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50, 80],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = 200
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False
