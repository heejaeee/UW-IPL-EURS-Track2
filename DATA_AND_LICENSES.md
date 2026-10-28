# Data and pretrained assets

This shared inventory covers both tracks and their inherited teacher/student training. Track-specific scene IDs and roles are recorded in FACTSHEET.json; a Track2-only entry does not imply use in Track1 inference. No dataset images are bundled. Repository/code licenses and dataset terms are distinguished.

| Asset | Source | License / evidence | Role |
|---|---|---|---|
| SAM3 | [source](https://github.com/facebookresearch/sam3) | SAM License, November 19, 2025; bundled actual LICENSE, not MIT despite pyproject classifier | Track1 frozen image/text model; Track2 logit and feature teacher |
| Restormer | [source](https://github.com/swz30/Restormer) | MIT | E034 label-conditioned pretrained SIDD real-denoising Restormer; smaller custom Restormer384 using the same layers |
| CascadePSP | [source](https://github.com/hkchengrex/CascadePSP) | MIT (bundled) | Track1 OTHER boundary refinement; NOT mixed into Track2 soft teacher targets |
| ConceptBank | [source](https://github.com/pgsmall/ConceptBank) | no license file in the public repository; used for academic research | Track1 train-only support/text fusion; retained in Track2 teacher |
| Torchvision MobileNetV2 ImageNet-1K | [source](https://pytorch.org/vision/stable/models/mobilenetv2.html) | Torchvision BSD-3-Clause code; ImageNet data terms separate | Track2 first 15 blocks, dilated OS8, LR-ASPP head |
| DINOv3-L/16 LVD-1689M | [model](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | [DINOv3 License (Meta)](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) | Track2 v4 training-only frozen dense-feature teacher, inherited through v5/v9 into v17 and the final student; not deployed; separate from v100 SAM3 feature KD |
| CoralscapesV2 TRAIN | [source](https://josauder.github.io/coralscapesv2/) | Apache-2.0 (official project page) | Track2 mapped segmentation GT, clean images and teacher completion; V2 extends the original Coralscapes dataset |
| SUIM TRAIN | [source](https://irvlab.cs.umn.edu/image-segmentation/suim) | used for non-commercial academic research | Track2 external supervision and teacher completion |
| UIIS10K TRAIN | [source](https://github.com/LiamLian0727/UIIS10K) | Apache-2.0 download LICENSE; upstream image rights may vary | E034 synthesis clean sources; inherited Track2 completion/scene expansion |
| USIS10K TRAIN | [source](https://github.com/LiamLian0727/USIS10K) | Apache-2.0 dataset card; upstream image rights may vary | E034 synthesis clean sources; inherited Track2 completion/scene expansion |
| AquaOV | [source](https://github.com/LiBingyu01/Earth2Ocean) | used for non-commercial academic research | Inherited Track2 background/partial-label external supervision; not in final 5,481-image logit KD pool |
| SUID: Synthetic Underwater Image Dataset | [IEEE DataPort](https://ieee-dataport.org/open-access/suid-synthetic-underwater-image-dataset); Hou et al., IEEE Access 2020; DOI [10.21227/agdr-y109](https://doi.org/10.21227/agdr-y109) | used for non-commercial academic research, in accordance with the dataset terms | 29 of the 30 clean ground-truth images (GTI) used for non-commercial academic research, in accordance with the dataset terms; Track2 restorer, alongside 1,789 CoralscapesV2 TRAIN clean images |
| Depth Anything V2 Small | [source](https://github.com/DepthAnything/Depth-Anything-V2) | Apache-2.0 (Small); no Base/Large model used in E034 synthesis | Offline E034 synthetic degradation; not inference |

The recorded Coralscapes downloads use `josauder/coralscapesV2` TRAIN, including v17 segmentation labels and contributor restoration sources. V2 extends the original dataset described by J. Sauder et al., *The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs*, 2025 ([arXiv:2503.20000](https://arxiv.org/abs/2503.20000)). No separate original-v1 dataset or v1 pretrained model use was found in the audited final lineage; the original citation credits dataset ancestry rather than a separate training pool.

SUID citation: G. Hou et al., *Benchmarking Underwater Image Enhancement and Restoration, and Beyond*, IEEE Access, vol. 8, pp. 122078–122091, 2020. Dataset DOI: [10.21227/agdr-y109](https://doi.org/10.21227/agdr-y109).
