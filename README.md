# VSViG: Real-time Video-based Seizure Detection via Skeleton-based Spatiotemporal ViG

This is official implementation of VSViG, here are some useful links:
[ [pdf](https://arxiv.org/pdf/2311.14775.pdf) | [data](https://huggingface.co/datasets/xuyankun/WU-SAHZU-EMU-Video) ]

**The code will be released soon!**

In this project, we propose a novel **V**ideo-based **S**eizure Detection via Skeleton-based Spatiotemporal **ViG** (**VSViG**) for efficient, accurate, and timely real-time automatic detection of epileptic seizures from surveillance videos .
Previous video-based seizure detection studies did not rely on skeleton information because the public pre-tained pose estimation model cannot track the patient poses accurately. We manually annotate the pose label for the patients, and train our own custom for patient pose estimation. 

Medical scenarios, specifically in EMUs  (Epilepsy Monitoring Units), are very complicated, public pre-trained pose estimation model cannot accurately track the patient skeletons, so we fine-tuned the custom pose estimation model with manual annotations for epileptic patients. The figure below shows the advantages of custom model, public pre-trained model cannot track all joints, and miss-track face keypoints.

<img src="https://github.com/xuyankun/VSViG/blob/main/compare.gif" width="550px">

You can test your own dataset (if your scenarios are very similar to ours, even patients with other movement-based diseases, e.g. PD) or fine-tune your own model with our provided custom pose estimation model [pose.pth](https://github.com/xuyankun/VSViG/blob/main/pose.pth). We utilize Openpose-lightweight to train the model, you can figure out how to implement it on your dataset in the original [Openpose-lightweight](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) project homepage. 

In this work, we propose a skeleton-based VSViG model with a partitioning strategy to recognize subtle seizure-related actions/behaviors, and generate probabilities rather than naive classification labels for timely detection. The figure below shows the performance of two seizure examples, $P$ stands for the real-time predictive probability/risk of seizure occurrence.

<img src="https://github.com/xuyankun/VSViG/blob/main/performance.gif" width="550px">



