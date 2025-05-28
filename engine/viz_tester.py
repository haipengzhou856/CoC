from tqdm import tqdm
from timm.utils.metrics import AverageMeter, accuracy
from torch.cuda.amp import autocast
import os
from accelerate.utils import set_seed
from solver.build_solver import build_optimizer, build_scheduler
from torch.nn import functional as F
import torch
import numpy as np
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from lifelines.statistics import logrank_test

def eval_func(cfg, accelerator, model, test_loader, logger, writer):
    device = accelerator.device
    patient_results = {}
    slide_ids = test_loader.dataset.slide_data['slide_id']
    all_risk_scores = np.zeros((len(test_loader)))
    all_censorships = np.zeros((len(test_loader)))
    all_event_times = np.zeros((len(test_loader)))
    idx = 0

    model = model.to(device)
    model.eval()
    for data in tqdm(test_loader, desc="Processing Testing batches"):
        slide_id = slide_ids.iloc[idx]

        wsi_features, wsi_global, omic_features, label, event_time, c = data
        wsi_features = wsi_features.to(device)
        wsi_global = wsi_global.to(device)
        omic_features = [omic.type(torch.FloatTensor).to(device) for omic in omic_features]
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        hazards, S = model(x_path=wsi_features[0],
                               x_global_img=wsi_global,
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

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[idx] = risk
        all_censorships[idx] = c.item()
        all_event_times[idx] = event_time

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(),
                                           'survival': event_time.item(), 'censorship': c.item()}})
        idx = idx + 1
    c_index = \
    concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    # ========== 修改后的Kaplan-Meier曲线部分 ==========
    # 1. 根据风险评分中位数分组
    median_risk = np.median(all_risk_scores)
    high_risk_group = (all_risk_scores > median_risk).astype(int)

    # 2. 准备生存分析数据
    event_observed = 1 - all_censorships.astype(int)

    # 3. 创建Kaplan-Meier拟合器
    kmf = KaplanMeierFitter()

    # 4. 创建图像
    fig = plt.figure(figsize=(10, 6), dpi=300)  # 提高DPI到300
    ax = fig.subplots()

    # 5. 绘制曲线
    for name, grouped_indices in [('Low Risk', ~high_risk_group.astype(bool)),
                                  ('High Risk', high_risk_group.astype(bool))]:
        if sum(grouped_indices) > 0:
            kmf.fit(all_event_times[grouped_indices],
                    event_observed=event_observed[grouped_indices],
                    label=name)
            kmf.plot_survival_function(ax=ax)

    # 6. 图表装饰
    ax.set_title(f'TCGA - {cfg.DATASET.CANCER_NAME}')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Survival Probability')
    #ax.grid(True)
    plt.tight_layout()  # 优化布局

    # ========== 新增p-value计算与显示 ==========
    # 计算log-rank检验的p值

    group1 = (high_risk_group == 0)
    group2 = (high_risk_group == 1)

    # 执行log-rank检验
    results = logrank_test(
        durations_A=all_event_times[group1],
        durations_B=all_event_times[group2],
        event_observed_A=event_observed[group1],
        event_observed_B=event_observed[group2]
    )

    # 格式化p值显示
    p_value = results.p_value



    # 7. 保存图片到当前目录
    filename = f"{cfg.DATASET.CANCER_NAME}_KM_curve.png"
    fig.savefig(
        filename,
        bbox_inches='tight',  # 防止标签被截断
        pad_inches=0.1,
        dpi=300
    )
    logger.info(f"Saved Kaplan-Meier curve to: {os.path.abspath(filename)}")
    plt.close(fig)  # 关闭图像释放内存

    return patient_results, c_index, p_value
