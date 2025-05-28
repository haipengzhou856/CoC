# CoC

**CoC: Chain-of-Cancer based on Cross-Modal Autoregressive Traction for Survival Prediction**

**MICCAI2025- Early Accepted**



Stay Tuned! We are on the way toward the clinical-related journal.



[update]-2025-02: Repo was created.

[update]-2025-05: This work has been early accepted by MICCAI25. 

[update]-2025-06: Raw Codes are uploaded, but not runnable. I'm on the way to tidy up :-)



## 1. Quick View

TODO



## 2. Data Preparation

### 2.1. ready for the raw data

Please refer to the [TCGA](https://portal.gdc.cancer.gov/) to download the raw data, including the raw WSIs (`.svs` files), Gene (`xls` files), and Methylation (`xls` files). You may also try to connect your collaborated doctors to check how to obtain and process these data. **Communicate with your doctor, I'm not expert for the data : )**

### 2.2. Embedding the WSIs

For WSIs, plz refer to [CLAM](https://github.com/mahmoodlab/CLAM) to check how to represent these giga-pixel images.

And you pathology data structure should look like:

```
Pathology
--BRCA
----masks
----patches
----ptfiles  # this is the vision embeddings
----stitiches
```

### 2.3. Data of the Gene&Methylation

We remain the original raw gene tabular data, which has `60660` features. I don't what they are since I'm not doctor  haha :) 

We transfer all the gene data into `XX.h5` .  For example in CESC, it looks like

|         | index | case_id | censorship | survival_days | slide_id | gene1data | ...  | gene60660data | meth1data | ...  | meth80000data |
| ------- | ----- | ------- | ---------- | ------------- | -------- | --------- | ---- | ------------- | --------- | ---- | ------------- |
| case1   |       |         |            |               |          |           |      |               |           |      |               |
| ...     |       |         |            |               |          |           |      |               |           |      |               |
| case277 |       |         |            |               |          |           |      |               |           |      |               |

Keep this design if you want use my code for reading data. 

## 3. Code Environments

**Note**: Since the resolution of WSI is various, and our model deploys CONCH. You should pay attention to your GPU memory if you want to reproduce our code. Our A6000 (48G) is approaching the OOM, plz watch out your GPUs for avoiding OOM. 

## 4. Usage of this Pipeline

TODO

## Note

-  I'm not familiar to the data acquisition, and the total data is very huge. **Thus, I will do not provide the ckpt weight.** So if you meet problem on our reproduction, plz post your issues here or contact to me.
-  Since the data could be various, thus I do not provide the `model weight` here. The overall data is very huge (2.5T for WSIs). **Again, for this task you should contact your doctor closely.** 
-  For other methods reproduction, please visit their official repo.  I thx [sicheng](https://script-yang.github.io/) who under my mentorship for providing most of these reproduced results.  

## TODO

- [ ] Release the full script for training & eval.
- [ ] Tidy up all the codes for directly running.

## Acknowledgement

* I do thanks for the talented RA sicheng&sihan (not brother lol, but they are XJTUer) who are under my mentorship for experiments and paper review. Big Thx!
* Thanks for the open-source researchers. 

## License

This project is under CC BY-NC 2.0, **Any kinds of modification is welcomed**.

But **Forbidden** Commercial Usage.

All Copyright **©** [Rydeen, Haipeng ZHOU](https://haipengzhou856.github.io/)

## Citation

If this paper and code are useful in your research, please consider citing:

```

```

