<h1 align="center">● SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation</h1>

<p align="center">
    <a href="https://discord.gg/DN4rvk95CC">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <img src="https://img.shields.io/static/v1?label=license&message=GPL&color=white&style=flat" alt="License"/>
</p>

SPA, is an advanced segmentation framework that efficiently adapts to diverse test-time preferences with minimal human interaction. By presenting users a select few, distinct segmentation candidates that best capture uncertainties, it reduces clinician workload in reaching the preferred segmentation. This method is elaborated on the paper [SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation](https://arxiv.org/abs/2411.15513) and [SPA webpage](https://supermedintel.github.io/SPA/). 

## 🔥 A Quick Overview 
 <div align="center"><img width="880" height="400" src="https://github.com/SuperMedIntel/SPA/blob/main/static/assets/images/facial.png"></div>
Our uncertainty-aware interactive segmentation model, <b>SPA</b>, efficiently achieves segmentations whose decisions on uncertain pixels are aligned with users preferences. This is achieved by modeling uncertainties and human interactions. At inference time, users are presented with one recommended prediction and a few representative segmentations that capture uncertainty, allowing users to select the one best aligned with their clinical needs. If the user is unsatisfied with the recommended prediction, the model learns from the users' selections, adapts itself, and presents users a new set of representative segmentations. Our approach minimizes user interactions and eliminates the need for painstaking pixel-wise adjustments compared to conventional interactive segmentation models.

## 🧐 Requirement

 Install the environment:

 ``conda env create -f environment.yml``

 ``conda activate SPA``

 Further Note: We tested on the following system environment and you may have to handle some issue due to system difference.
```
Operating System: Ubuntu 22.04
Conda Version: 23.7.4
Python Version: 3.12.4
```

## 🎯 Example Cases
**Step1:** Download MedSAM or SAM pretrained weight and put in the ``./checkpoint/sam/`` folder, create the folder if it does not exist ⚒️
 [MedSAM checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link)
 
 [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

**Step2:** Download REFUGE2 (update later) or your own multi-rater dataset and put in the ``data`` folder, create the folder if it does not exist ⚒️
 
**Step3:** Run the training by:
 ``python train.py -net sam -mod sam -exp_name 'REFUGE_SPA' -sam_ckpt ./checkpoint/sam/medsam_vit_b.pth -image_size 512 -out_size 256 -b 4 -val_freq 1 -dataset REFUGE -data_path './data/REFUGE'``

**Step4:** Run the validation by:
 ``python val.py -net sam -mod sam -exp_name 'val' -vis 1 -sam_ckpt CHECKPOINT_PATH -weights CHECKPOINT_PATH -image_size 512 -out_size 256 -b 1 -val_freq 1 -dataset REFUGE -data_path './data/REFUGE'``

## 🚨 News
- 25-06-26. SPA is accepted by ICCV 🥳
- 25-01-06. Code Uploaded 👩‍💻
- 24-12-02. SPA's website is released 🤩

## 📝 Cite
 ~~~
@misc{zhu_spa_2024,
      title={SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation},
      author={Zhu, Jiayuan and Wu, Junde and Ouyang, Cheng and Kamnitsas, Konstantinos and Noble, Alison},
      url = {http://arxiv.org/abs/2411.15513},
      doi = {10.48550/arXiv.2411.15513},
      year = {2024},
    }
 ~~~
