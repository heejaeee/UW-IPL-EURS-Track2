# Data and pretrained assets

See FACTSHEET.json for scene IDs and per-asset roles. Unknown licenses below are explicitly unresolved; no dataset images are bundled.

| Asset | Source | License / evidence | Role |
|---|---|---|---|
| SAM3 | [source](https://github.com/facebookresearch/sam3) | SAM License, November19 2025; bundled actual LICENSE, not MIT despite pyproject classifier | Track1 frozen image/text model; Track2 logit and feature teacher |
| Restormer | [source](https://github.com/swz30/Restormer) | MIT | E034 label-conditioned pretrained SIDD real-denoising Restormer; smaller custom Restormer384 using the same layers |
| CascadePSP | [source](https://github.com/hkchengrex/CascadePSP) | MIT (bundled) | Track1 OTHER boundary refinement; NOT mixed into Track2 soft teacher targets |
| ConceptBank | [source](https://github.com/pgsmall/ConceptBank) | No LICENSE file found in pinned checkout; licensing not verified. Source attribution retained; confirm redistribution terms before public release. | Track1 train-only support/text fusion; retained in Track2 teacher |
| Torchvision MobileNetV2 ImageNet-1K | [source](https://pytorch.org/vision/stable/models/mobilenetv2.html) | Torchvision BSD-3-Clause code; ImageNet data terms separate | Track2 first15blocks, dilated OS8, LR-ASPP head |
| CoralscapesV2 TRAIN | [source](https://josauder.github.io/coralscapesv2/) | Apache-2.0 (official project page) | Track2 mapped segmentation GT, clean images and teacher completion |
| SUIM TRAIN | [source](https://irvlab.cs.umn.edu/image-segmentation/suim) | Dataset rights not independently verified; do not substitute SUIM-Net code license | Track2 external supervision and teacher completion |
| UIIS10K TRAIN | [source](https://github.com/LiamLian0727/UIIS10K) | Apache-2.0 download LICENSE; upstream image rights may vary | E034 synthesis clean sources; inherited Track2 completion/scene expansion |
| USIS10K TRAIN | [source](https://github.com/LiamLian0727/USIS10K) | Apache-2.0 dataset card; upstream image rights may vary | E034 synthesis clean sources; inherited Track2 completion/scene expansion |
| AquaOV | [source](https://github.com/LiBingyu01/Earth2Ocean) | Dataset license not verified | Inherited Track2 background/partial-label external supervision; not in final5481logitKD pool |
| SUID clean images | Unconfirmed contributor source | Contributor-provided; source URL and dataset license still require confirmation | Team Track2 restorer:29cleanimages, plus1789CoralscapesV2TRAIN |
| Depth Anything V2 Small | [source](https://github.com/DepthAnything/Depth-Anything-V2) | Apache-2.0 (Small); no Base/Large model used in E034 synthesis | Offline E034 synthetic degradation; not inference |
