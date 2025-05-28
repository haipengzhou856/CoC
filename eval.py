import os
import argparse
from configs import get_cfg_default
import accelerate
from accelerate.utils import set_seed
import clip
from utils.utils import SetEverything, cal_param,load_module_from_path
import torch.distributed as dist
from datareader.readerv1 import Generic_WSI_Survival_Dataset,Generic_Split,Generic_MIL_Survival_Dataset
from torch.utils.data import DataLoader
import numpy as np
from engine.viz_tester import eval_func

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, help='config of the dataset')
    parser.add_argument('--model_config',type=str, help='config of the model')
    parser.add_argument('--data_mode', type=str,default='coattn', help='mode of data retrieval')
    parser.add_argument('--exp_name', type=str, default='debug',help='name your experiment')
    args = parser.parse_args()
    # freeze the config
    cfg = get_cfg_default()
    cfg.EXP_NAME = args.exp_name
    cfg.INPUT.MODE = args.data_mode
    cfg.merge_from_file(args.dataset_config)          # yaml file
    module = load_module_from_path(args.model_config) # py file
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

    resume_path = cfg.RESUME_DIR


    #############################################################################
    ################################  Log basics ################################
    #############################################################################
    logger.info("----------------------NEW RUN----------------------------")

    #############################################################################
    ################################  get dataloader ############################
    #############################################################################
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
    fold_names = ['splits_2.csv']
    for i_fold in fold_names:
        logger.info("--------------Run {} fold----------------".format(i_fold))
        split_path = os.path.join(cfg.DATASET.ROOT, dataset_name,i_fold)
        train_dataset, val_dataset = overall_dataset.return_splits(
                                        from_id=False,
                                        csv_path=split_path)
        dataloaders = {}
        train_loader = DataLoader(train_dataset,
                            batch_size=cfg.DATALOADER.BATCH_SIZE,
                            shuffle=True,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            pin_memory=False)
        test_loader = DataLoader(val_dataset,
                            batch_size=cfg.DATALOADER.BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            pin_memory=False)

        # TODO
        logger.info("-----------------Finish dataloader----------------")

        from model import SNN, CMTA, CoC_V2
        model = CoC_V2()
        logger.info("-------------- Finish creating model --------------")

        # load model checkpoint
        logger.info(f"Resumed from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)

        #############################################################################
        ################################  eval scheme  ##############################
        #############################################################################

        patient_results, cur_cidx,p_value = eval_func(cfg, accelerator, model, test_loader, logger, writer)

    logger.info("-------------- C-index:{} --------------".format(cur_cidx))
    logger.info("-------------- p-value:{} --------------".format(p_value))
    logger.info("-----------------End RUN!----------------")

