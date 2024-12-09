# VSViG: Real-time Video-based Seizure Detection via Skeleton-based Spatiotemporal ViG

This is the official implementation of VSViG, which is accepted by ECCV 2024.

> Paper:  <a href="https://arxiv.org/pdf/2311.14775.pdf"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>

> Dataset:  [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/xuyankun/WU-SAHZU-EMU-Video)

<img src="https://github.com/xuyankun/VSViG/blob/main/performance.gif" width="550px">

In this project, we propose a novel **V**ideo-based **S**eizure Detection via Skeleton-based Spatiotemporal **ViG** (**VSViG**) for efficient, accurate, and timely real-time automatic detection of epileptic seizures from surveillance videos .
Previous video-based seizure detection studies did not rely on skeleton information because the public pre-tained pose estimation model cannot track the patient poses accurately. We manually annotate the pose label for the patients, and train our own custom for patient pose estimation. 

Medical scenarios, specifically in EMUs  (Epilepsy Monitoring Units), are very complicated, public pre-trained pose estimation model cannot accurately track the patient skeletons, so we fine-tuned the custom pose estimation model with manual annotations for epileptic patients. The figure below shows the advantages of custom model, public pre-trained model cannot track all joints, and miss-track face keypoints.

<img src="https://github.com/xuyankun/VSViG/blob/main/compare.gif" width="550px">

You can test your own dataset (if your scenarios are very similar to ours, even patients with other movement-based diseases, e.g. PD) or fine-tune your own model with our provided custom pose estimation model [pose.pth](https://github.com/xuyankun/VSViG/blob/main/pose.pth). We utilize Openpose-lightweight to train the model, you can figure out how to implement it on your dataset in the original [Openpose-lightweight](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) project homepage. 

## VSViG Implementation
In this work, we propose a skeleton-based VSViG model with a partitioning strategy to recognize subtle seizure-related actions/behaviors, and generate probabilities rather than naive classification labels for timely detection. The figure below shows the performance of two seizure examples, $P$ stands for the real-time predictive probability/risk of seizure occurrence.

We have provided [VSViG model](https://github.com/xuyankun/VSViG/blob/main/VSViG.py) and its [train.py](https://github.com/xuyankun/VSViG/blob/main/train.py) code in this repository, and uploaded utilized [dataset](https://huggingface.co/datasets/xuyankun/WU-SAHZU-EMU-Video). We just provided raw video recordings with onset annotation, if you want to utilize our dataset, you have to preprocess them into video clips for training. It is worth noting that, due to rules from the hospital, we have to protect patients' privacy by masking faces, so that you cannot recognize subtle changes on their faces during the beginning phase of seizure onset. The provided annotation is based on real scenarios, so that you might achieve worse latency performance than the results in our paper. 

The label for each clip should be in a probabilistic likelihood, and we recommend to label the clip as:
```
[Health, EEG onset]: 0
[Clinical onset, ]: 1
[EEG onset, Clinical onset]: 0 → 1 in an exponential way
```

We also provided the patch extraction operation [here](https://github.com/xuyankun/VSViG/blob/main/extract_patches.py), we simplified the operation by generating a patch with gaussian kernel based on each keypoint. We randomly generated [dynamic_partition_order](https://github.com/xuyankun/VSViG/blob/main/dy_point_order.pt) during the training phase, that has been proven to increase the performance. If you want to test our model with shuffled partition module, just add this file to the model. If you wan to train your own model, just randomly generate a new order file, and keep it same in model inference.

You can either train your own model or test our pre-trained [VSViG-base.pth](https://github.com/xuyankun/VSViG/blob/main/VSViG-base.pth) on custom or our dataset.


## Citation:

If this paper or dataset helps your research, please cite the paper:

```
@inproceedings{xu2024vsvig,
  title={VSViG: Real-Time Video-Based Seizure Detection via Skeleton-Based Spatiotemporal ViG},
  author={Xu, Yankun and Wang, Junzhe and Chen, Yun-Hsuan and Yang, Jie and Ming, Wenjie and Wang, Shuang and Sawan, Mohamad},
  booktitle={European Conference on Computer Vision},
  pages={228--245},
  year={2024},
  organization={Springer}
}

@inproceedings{Xu2023VSViG,
  title={VSViG: Real-time Video-based Seizure Detection via Skeleton-based Spatiotemporal ViG},
  author={Yankun Xu and Junzhe Wang and Yun-Hsuan Chen and Jie Yang and Wenjie Ming and Shuangquan Wang and Mohamad Sawan},
  booktitle={arXiv preprint arXiv:2311.14775},
  year={2023}
}
```

We also recommend our previous paper about EEG-based early seizure detection, which first proposed the probabilistic prediction concept for seizure detection task, also proposed rectified weighting strategy and decision-making rule can also enhance the detection latency performance.
```
@article{xu2024shorter,
  title={Shorter latency of real-time epileptic seizure detection via probabilistic prediction},
  author={Xu, Yankun and Yang, Jie and Ming, Wenjie and Wang, Shuang and Sawan, Mohamad},
  journal={Expert Systems with Applications},
  volume={236},
  pages={121359},
  year={2024},
  publisher={Elsevier}
}
```


