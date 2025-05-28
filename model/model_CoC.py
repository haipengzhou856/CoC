# Implementation of Chain-of-Cancer model
# All rights reserved by Rydeen
# Only Academic use is allowed
# 2025.2


import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import *
import numpy as np
from .model_CMTA import TransLayer
from .MAR import AutoregressiveDiffusion, MAR
from .conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

CLIP, _ = create_model_from_pretrained('conch_ViT-B-16', "/home/haipeng/Pretrained/CONCH/pytorch_model.bin")
Tokenizer = get_tokenizer()


class WSIProjector(nn.Module):
    def __init__(self, num_layers, num_token, dim=512):
        super().__init__()
        self.dim = dim
        self.num_cls_token = num_token
        self.cls_tokens = nn.Parameter(torch.randn(1, self.num_cls_token, dim))
        self.layers = nn.ModuleList([TransLayer(dim=dim, norm_layer=nn.LayerNorm) for _ in range(num_layers)])

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for layer in self.layers:
            x = layer(x)
        return x[:, :self.num_cls_token]



class CoC_V2(nn.Module):
    def __init__(self, gene_sizes=[10110, 10110, 10110, 10110, 10110, 10110],
                 meth_size=[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                 n_classes=4, cancer="BRCA"):
        super(CoC_V2, self).__init__()
        self.n_classes = n_classes

        # TODO
        self.size_dict = {
            "path": [1024, 512, 512],
            "gene": [1024, 1024, 512, 512],
            "meth": [1024, 1024, 512, 512],
        }
        self.latent_dim = 512
        self.num_proj_token = 12
        self.cancer = cancer
        #########################################################################
        ######################### Raw Feature  Network  #########################
        # SNN of gene
        snn_gene = []
        for input_dim in gene_sizes:
            fc = [SNN_Block(dim1=input_dim, dim2=self.size_dict["gene"][0])]
            for i, _ in enumerate(self.size_dict["gene"][1:]):
                fc.append(SNN_Block(dim1=self.size_dict["gene"][i], dim2=self.size_dict["gene"][i + 1],
                                    dropout=0.25))
            snn_gene.append(nn.Sequential(*fc))
        self.snn_gene = nn.ModuleList(snn_gene)

        # SNN of meth
        snn_meth = []
        for input_dim in meth_size:
            fc = [SNN_Block(dim1=input_dim, dim2=self.size_dict["meth"][0])]
            for i, _ in enumerate(self.size_dict["meth"][1:]):
                fc.append(SNN_Block(dim1=self.size_dict["meth"][i], dim2=self.size_dict["meth"][i + 1],
                                    dropout=0.25))
            snn_meth.append(nn.Sequential(*fc))
        self.snn_meth = nn.ModuleList(snn_meth)

        ## projector of wsi
        fc = []
        for idx in range(len(self.size_dict["path"]) - 1):
            fc.append(nn.Linear(self.size_dict["path"][idx], self.size_dict["path"][idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.fc_wsi = nn.Sequential(*fc)

        ########################################################################
        ######################### CoC Traction Network #########################

        self.clip = CLIP
        self.Tokenizer = Tokenizer
        self.init_text = ['an H&E stained image of {cancer}'.format(cancer=cancer)]

        self.CrossAttnGlobal = MultiheadAttention(embed_dim=self.latent_dim, num_heads=8)
        self.CrossAttnLocal = MultiheadAttention(embed_dim=self.latent_dim, num_heads=8)

        ## WSI projector
        self.WSIProjector = WSIProjector(num_layers=2, num_token=self.num_proj_token, dim=self.latent_dim)

        self.adapter_gene = nn.Sequential(
            *[nn.Linear(self.latent_dim * 7, self.latent_dim),
              nn.ReLU(),
              nn.Linear(self.latent_dim, self.latent_dim),
              nn.ReLU()]
        )
        self.adapter_meth = nn.Sequential(
            *[nn.Linear(self.latent_dim * 9, self.latent_dim),
              nn.ReLU(),
              nn.Linear(self.latent_dim, self.latent_dim),
              nn.ReLU()]
        )

        ########################################################################
        ################################ decoder ###############################
        self.intra_head = nn.Sequential(
            *[nn.Linear(self.latent_dim * 27, self.latent_dim),
              nn.ReLU(),
              nn.Linear(self.latent_dim, self.latent_dim),
              nn.ReLU()]
        )
        self.inter_head = nn.Sequential(
            *[nn.Linear(self.latent_dim * 4, self.latent_dim),
              nn.ReLU(),
              nn.Linear(self.latent_dim, self.latent_dim),
              nn.ReLU()]
        )

        # TODO
        self.fuse = nn.Sequential(
            *[nn.Linear(self.latent_dim, self.latent_dim),
              nn.ReLU(),
              nn.Linear(self.latent_dim, self.latent_dim),
              nn.ReLU()]
        )
        self.classifier = nn.Linear(self.latent_dim, self.n_classes)

        self.apply(initialize_weights)
        self.MAR = MAR(embed_size=self.latent_dim,
                       num_layers=2,
                       nhead=4, )


    def forward(self, **kwargs):
        feature_gene, feature_meth, feature_wsi_l, feature_wsi_g = self.feature_engineering(**kwargs)
        device = feature_gene.device
        ########################################## CoC  #####################################
        #####################################################################################

        ## CoC-1: visual-Global prompt, in a hand-crafted way
        CoC_global_prompt = [
            "an H&E stained image of {} at Slide-Level. Focus on the Tumor Boundaries, Tissue Structure, and Inflammatory Response".format(
                self.cancer)]
        CoC_global_prompt = tokenize(texts=CoC_global_prompt, tokenizer=self.Tokenizer).to(device)

        ## CoC-2: visual-Local prompt, in a hand-crafted way
        CoC_local_prompt = [
            "an H&E stained image of {} at Patch-Level. Focus on the Cell Morphology, Intercellular Bridges, and Vascular and Lymphatic Invasion".format(
                self.cancer)]
        CoC_local_prompt = tokenize(texts=CoC_local_prompt, tokenizer=self.Tokenizer).to(device)

        with torch.no_grad():
            CoC_global_prompt = self.clip.encode_text(CoC_global_prompt).unsqueeze(0)
            CoC_local_prompt = self.clip.encode_text(CoC_local_prompt).unsqueeze(0)

        # Detach the tensors to make them normal tensors for autograd
        CoC_global_prompt = CoC_global_prompt.clone()
        CoC_local_prompt = CoC_local_prompt.clone()

        # Global
        feature_wsi_g = feature_wsi_g.unsqueeze(0).clone()
        GlobalToken, _ = self.CrossAttnGlobal(CoC_global_prompt, feature_wsi_g,
                                              feature_wsi_g)  # [1,1,512]# TODO: cross attention,
        # Local
        LocalWSIToken = self.WSIProjector(feature_wsi_l)  # [1,num_patches,512] -> [1,12,512], compression first
        LocalToken, _ = self.CrossAttnLocal(CoC_local_prompt, LocalWSIToken,
                                            LocalWSIToken)  # [1,1,512]# TODO: cross attention,



        ## 3. CoC-2: Omics prompt: ["an H&E stained image of {cancer}"]
        gene_prompt = ["an H&E stained image of {cancer} with abnormal gene expression".format(cancer=self.cancer)]
        meth_prompt = ["an H&E stained image of {cancer} with abnormal methylation".format(cancer=self.cancer)]
        gene_prompt = tokenize(texts=gene_prompt, tokenizer=self.Tokenizer).to(device)
        meth_prompt = tokenize(texts=meth_prompt, tokenizer=self.Tokenizer).to(device)
        with torch.no_grad():
            gene_prompt = self.clip.encode_text(gene_prompt).unsqueeze(0)
            meth_prompt = self.clip.encode_text(meth_prompt).unsqueeze(0)
        gene_prompt = torch.cat((gene_prompt, feature_gene), dim=1)  # [1,1+6,512]
        meth_prompt = torch.cat((meth_prompt, feature_meth), dim=1)  # [1,1+8,512]

        embed_gene = self.adapter_gene(gene_prompt.flatten(1))
        embed_gene = embed_gene.unsqueeze(0)
        embed_meth = self.adapter_meth(meth_prompt.flatten(1))
        embed_meth = embed_meth.unsqueeze(0)

        ## MAR in training
        pred, aux_loss = self.MAR(GlobalToken, LocalToken, embed_gene, embed_meth)

        intra_fea = self.intra_head(
            torch.cat([LocalWSIToken, feature_wsi_g, feature_gene, feature_meth], dim=1).flatten(1))
        inter_fea = self.inter_head(pred.flatten(1))

        # TODO
        fusion = self.fuse(inter_fea+intra_fea)
        #fusion = self.fuse(torch.cat([inter_fea, intra_fea], dim=1))
        # TODO: fusion is the traction of all the modal data
        logits = self.classifier(fusion)  # [1, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        if self.training:
            return hazards, S, aux_loss
        else:
            return hazards, S,

    def feature_engineering(self, **kwargs):
        '''
        :param kwargs:
        :return:
        feature_gene: [1, 6, 512]
        feature_meth: [1, 8, 512]
        feature_wsi_l: [1, instance_num, 512], locally
        feature_wsi_g: [1, 512], globally

        these features are raw and heterogeneous
        '''
        device = kwargs["x_path"].device
        ###############  basic feature extraction ##############
        x_path = kwargs["x_path"]  # [instance_num, dim]
        x_gene = [kwargs["x_gene%d" % i] for i in range(1, 7)]  # list of [[gene_dim],...]
        x_meth = [kwargs["x_meth%d" % i] for i in range(1, 9)]  # list of [[meth_dim],...]
        x_global_img = kwargs["x_global_img"]  # [1,3,448,448]

        feature_gene = [self.snn_gene[idx].forward(x)
                        for idx, x in enumerate(x_gene)]
        feature_gene = torch.stack(feature_gene).unsqueeze(0)  # [1, 6, 512]
        # feature_gene = feature_gene.flatten(1)

        feature_meth = [self.snn_meth[idx].forward(x)
                        for idx, x in enumerate(x_meth)]
        feature_meth = torch.stack(feature_meth).unsqueeze(0)  # [1, 8, 512]
        # feature_meth = feature_meth.flatten(1)

        feature_wsi_l = self.fc_wsi(x_path).unsqueeze(0)  # [1, instance_num, 512], locally
        # TODO
        with torch.inference_mode():
            feature_wsi_g = self.clip.encode_image(x_global_img)  # [1,512]
        return feature_gene, feature_meth, feature_wsi_l, feature_wsi_g
