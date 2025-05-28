from yacs.config import CfgNode as CN

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    cfg.MODEL.CMTA = CN()
    cfg.MODEL.CMTA.OMIC_DIM = [10110,10110,10110,10110,10110,10110]



