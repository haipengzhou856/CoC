############## steal this config style from Dassl and Detectron2 #############
######################### big thanks to the authors ##########################

from yacs.config import CfgNode as CN

#####################  Env config #####################
_C = CN()
_C.SEED = 42
_C.OUTPUT_DIR = './output'
_C.LOG = '/log_path'  # logs
_C.TB = '/tb_path'  # tensorboard
_C.CKPT = '/ckpt_path'  # checkpoints
_C.EXP_NAME = 'debug'
_C.RESUME_DIR = '/home/haipeng/MICCAI25/Path/output/CoC-KIRC-v2/2025-02-13-18-03-33/ckpt_path/ckpt_epoch20'

#####################  Input config #####################
_C.INPUT = CN()
_C.INPUT.GENE_DIM = 60660
_C.INPUT.METH_DIM = 80000
_C.INPUT.MODE = 'meth' # 'gene', 'meth', 'omic', 'pathomic','coattn'



#####################  Dataset config #####################
_C.DATASET = CN()
_C.DATASET.ROOT = "/vip_media/SharedData/SurvivalPred/data"
_C.DATASET.CANCER_NAME = ""
_C.DATASET.LOCAL_WSI_PATH = "WSI/pt_files"  # pathology pt files from CLAM
_C.DATASET.GLOBAL_WSI_PATH = "masks"  # pathology pt files from CLAM
_C.DATASET.GENE_PATH = "Gene/gene.h5" # gene data, we mannually preprocess and convert it to h5 file
_C.DATASET.METH_PATH = "Meth/meth.h5" # methylation data, we mannually preprocess and convert it to h5 file
_C.DATASET.OMIC_PATH = "Omic/omic_data.h5" # methylation data, we mannually preprocess and convert it to h5 file





#####################  Dataloader config #####################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 2
_C.DATALOADER.BATCH_SIZE = 1



#####################  Model config #####################
_C.MODEL = CN()
_C.MODEL.CoC = CN()
_C.MODEL.SNN = CN()
_C.MODEL.CMTA = CN()
_C.MODEL.MCAT = CN()


#####################  Solver config #####################
_C.SOLVER = CN()
_C.SOLVER.LR = 0.00001
_C.SOLVER.EPOCH = 20
_C.SOLVER.STORE_EPOCH = 1
_C.SOLVER.OPTIMIZER = 'AdamW'
_C.SOLVER.SGD_MOMENTUM = 0.9
_C.SOLVER.DECAY = 0.001

_C.SOLVER.SCHEDULER = 'Linear'
_C.SOLVER.STEP_GAMMA=0.8
