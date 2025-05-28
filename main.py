import os
import argparse
from configs import get_cfg_default
import accelerate
from accelerate.utils import set_seed
import clip
from utils.utils import SetEverything, cal_param, load_module_from_path
import torch.distributed as dist
from datareader.readerv1 import Generic_WSI_Survival_Dataset, Generic_Split, Generic_MIL_Survival_Dataset
from torch.utils.data import DataLoader
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, help='config of the dataset')
    parser.add_argument('--model_config', type=str, help='config of the model')
    parser.add_argument('--data_mode', type=str, default='coattn', help='mode of data retrieval')
    parser.add_argument('--exp_name', type=str, default='debug', help='name your experiment')
    args = parser.parse_args()
    # freeze the config
    cfg = get_cfg_default()
    cfg.EXP_NAME = args.exp_name
    cfg.INPUT.MODE = args.data_mode
    cfg.merge_from_file(args.dataset_config)  # yaml file
    module = load_module_from_path(args.model_config)  # py file
    assert hasattr(module, 'extend_cfg'), "Check the model config file, extend_cfg function is not found"
    module.extend_cfg(cfg)
    cfg.freeze()

    #############################################################################
    ################################  Set miscellaneous #########################
    #############################################################################
    # NOTE: Only support for batch size 1, thus no need to deploy DDP

    # set accelerator
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    # set everything
    logger, writer, ckpt_path, out_dir = SetEverything(cfg, accelerator.process_index)
    set_seed(cfg.SEED)

    #############################################################################
    ################################  Log basics ################################
    #############################################################################
    logger.info("----------------------NEW RUN----------------------------")
    logger.info("-------------------Basic Setting-------------------------")
    logger.info("---output in: {dir}---".format(dir=out_dir))
    logger.info("BATCH_SIZE: {}".format(cfg.DATALOADER.BATCH_SIZE))
    logger.info("lr: {}".format(cfg.SOLVER.LR))
    logger.info("opim: {}".format(cfg.SOLVER.OPTIMIZER))
    logger.info(
        "Using {num_gpu} GPU for training, {mix_pix} used.".format(num_gpu=accelerator.num_processes,
                                                                   mix_pix=accelerator.mixed_precision))

    #############################################################################
    ################################  get dataloader ############################
    #############################################################################
    dataset_name = cfg.DATASET.CANCER_NAME
    # gene_path = os.path.join(cfg.DATASET.ROOT, dataset_name,
    #                         cfg.DATASET.GENE_PATH)
    omic_path = os.path.join(cfg.DATASET.ROOT, dataset_name,
                             cfg.DATASET.OMIC_PATH)
    local_wsi_path = os.path.join(cfg.DATASET.ROOT, dataset_name,
                                  cfg.DATASET.LOCAL_WSI_PATH)
    # meth_path = os.path.join(cfg.DATASET.ROOT, dataset_name,
    #                         cfg.DATASET.METH_PATH) # TODO
    overall_dataset = Generic_MIL_Survival_Dataset(h5_path=omic_path,
                                                   data_dir=local_wsi_path,
                                                   mode=cfg.INPUT.MODE)

    ############################### run 5-fold cross-validation ###############################
    fold_names = [ 'splits_3.csv', 'splits_4.csv','splits_0.csv', 'splits_1.csv', 'splits_2.csv']
    # fold_names = [ 'splits_3.csv', 'splits_4.csv']
    results_cindex = []
    for i_fold in fold_names:
        logger.info("--------------Run {} fold----------------".format(i_fold))
        split_path = os.path.join(cfg.DATASET.ROOT, dataset_name, i_fold)
        train_dataset, val_dataset = overall_dataset.return_splits(
            from_id=False,
            csv_path=split_path)
        dataloaders = {}
        dataloaders['train'] = DataLoader(train_dataset,
                                          batch_size=cfg.DATALOADER.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=cfg.DATALOADER.NUM_WORKERS,
                                          pin_memory=False)
        dataloaders['test'] = DataLoader(val_dataset,
                                         batch_size=cfg.DATALOADER.BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=cfg.DATALOADER.NUM_WORKERS,
                                         pin_memory=False)
        # TODO
        logger.info("-----------------Finish dataloader----------------")

        #############################################################################
        ################################  create model ##############################
        #############################################################################
        from model import SNN, CMTA, CoC_V2

        # model = SNN(omic_input_dim=cfg.MODEL.SNN.OMIC_DIM,model_size_omic='big')
        # model = CMTA(omic_sizes=cfg.MODEL.CMTA.OMIC_DIM)
        model = CoC_V2()
        logger.info("--------------Finish creating model----------------")

        #############################################################################
        ################################  train scheme ##############################
        #############################################################################
        # from engine.trainer import training_func
        # training_func(cfg, accelerator, model, dataloaders, logger, writer, ckpt_path)

        from engine.trainer import training_func

        tmp_result = training_func(cfg, accelerator, model, dataloaders, logger, writer, ckpt_path, prefix=i_fold)
        results_cindex.append(tmp_result)

    results_cindex = np.array(results_cindex)
    logger.info("{} cancer of Avg C-Index {:.3f}, stdp: {:.3f}, stds: {:.3f}".format(
        dataset_name, results_cindex.mean(), results_cindex.std(), results_cindex.std(ddof=1)))
    logger.info("-----------------End RUN!----------------")
