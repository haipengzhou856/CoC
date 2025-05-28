from tqdm import tqdm
from timm.utils.metrics import AverageMeter, accuracy
from torch.cuda.amp import autocast
import os
from accelerate.utils import set_seed
from solver.build_solver import build_optimizer, build_scheduler
from solver.loss_func import *
from torch.nn import functional as F
from .tester import eval_func
import numpy as np
from sksurv.metrics import concordance_index_censored


def training_func(cfg, accelerator, model, dataloaders, logger, writer, ckpt_path,prefix):
    set_seed(cfg.SEED)
    loss_fn = NLLSurvLoss(alpha=0)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    train_loader, test_loader = dataloaders['train'], dataloaders['test']

    total_epoch = cfg.SOLVER.EPOCH
    store_epoch = cfg.SOLVER.STORE_EPOCH
    device = accelerator.device
    #resume_path = cfg.RESUME_DIR # ONLY USED FOR EMERGENCY SITUATION, NEED MANUALLY CHANGE CONFIG

    model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, test_loader)

    logger.info("----------------Starting training------------------")
    logger.info(f"--------------Total {total_epoch} Epochs--------------")

    '''accelerator.print(resume_path)
    if resume_path != "":  # if resume
        logger.info(f"Resumed from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)
        path = os.path.basename(resume_path)
        starting_epoch = int(path.replace("ckpt_epoch", "")) + 1
    else:
        starting_epoch = 1'''

    overall_step = 0
    best_cidx = 0
    best_epoch = 0

    for epoch in range(1, total_epoch+1):
        all_risk_scores = np.zeros((len(train_loader)))
        all_censorships = np.zeros((len(train_loader)))
        all_event_times = np.zeros((len(train_loader)))
        idx = 0
        model.train()
        for data in tqdm(train_loader, desc=f"Processing Training batches with {total_epoch}".format(total_epoch)):
            wsi_features,wsi_global, omic_features, label, event_time, c = data
            omic_features = [omic.type(torch.FloatTensor).to(device) for omic in omic_features]
            label = label.type(torch.LongTensor).to(device)
            c = c.type(torch.FloatTensor).to(device)

            if cfg.INPUT.MODE == "gene" or cfg.INPUT.MODE == "meth":
                hazards, S, Y_hat, A, _ = model(x_omic=omic_features[0])
            elif cfg.INPUT.MODE == "coattn":
                hazards, S, _,_,_,_ = model(x_path=wsi_features[0],
                                                x_omic1=omic_features[0][0],
                                                x_omic2=omic_features[1][0],
                                                x_omic3=omic_features[2][0],
                                                x_omic4=omic_features[3][0],
                                                x_omic5=omic_features[4][0],
                                                x_omic6=omic_features[5][0]
                                                )
            elif cfg.INPUT.MODE == "coc":
                hazards, S,aux_loss = model(x_path=wsi_features[0],
                                   x_global_img = wsi_global,
                                               x_gene1=omic_features[0][0],
                                               x_gene2=omic_features[1][0],
                                               x_gene3=omic_features[2][0],
                                               x_gene4=omic_features[3][0],
                                               x_gene5=omic_features[4][0],
                                               x_gene6=omic_features[5][0],
                                               x_meth1=omic_features[6][0],
                                               x_meth2=omic_features[7][0],
                                               x_meth3=omic_features[8][0],
                                               x_meth4=omic_features[9][0],
                                               x_meth5=omic_features[10][0],
                                               x_meth6=omic_features[11][0],
                                               x_meth7=omic_features[12][0],
                                               x_meth8=omic_features[13][0],
                                               )

            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c) + aux_loss
            accelerator.backward(loss)
            ################# update optimizer #################
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            overall_step += 1
            ###################### monitor #####################
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[idx] = risk
            all_censorships[idx] = c.item()
            all_event_times[idx] = event_time
            writer.add_scalar("{}-loss".format(prefix.split('.')[0]), loss.item(), overall_step)
            writer.add_scalar("{}-lr".format(prefix.split('.')[0]), optimizer.state_dict()['param_groups'][0]['lr'], overall_step)
            idx = idx + 1

        scheduler.step(epoch)
        c_index_train = \
            concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                       tied_tol=1e-08)[0]
        writer.add_scalar('{}-train/c_index'.format(prefix.split('.')[0]), c_index_train, epoch)

        if epoch % store_epoch == 0:
            logger.info(f"----------------Save ckpt_epoch{epoch}------------------")
            restore_path = os.path.join(ckpt_path, f"ckpt_epoch{epoch}")
            #accelerator.save_state(restore_path)  # TODO: bugs when using Poly scheduler and cant to save
            patient_results, cur_cidx = eval_func(cfg,accelerator, model, test_loader, epoch)
            writer.add_scalar('{}-val/c_index'.format(prefix.split('.')[0]), cur_cidx, epoch)
            if cur_cidx > best_cidx:
                best_cidx = cur_cidx
                best_epoch = epoch
            logger.info(f"******Epoch {epoch} C-Index: {cur_cidx}---------------")
            logger.info(f"******CURRENT BEST C-Index: {best_cidx}---------------")
            logger.info(f"******CURRENT BEST EPOCH: {best_epoch}---------------")
    # TODO
    return best_cidx
