# STViG-for-READS-V
**The code will be released soon!**

In this project, we propose a novel skeleton-based **S**patio**T**emporal **Vi**sion **G**raph neural network (**STViG**) for efficient, accurate, and timely **RE**al-time **A**utomated **D**etection of epileptic **S**eizures from surveillance **V**ideos (**READS-V**).
Previous video-based seizure detection studies did not rely on skeleton information because the public pre-tained pose estimation model cannot track the patient poses accurately. We manually annotate the pose label for the patients, and train our own custom for patient pose estimation. 

Medical scenarios, specifically in EMUs  (Epilepsy Monitoring Units), are very complicated, public pre-trained pose estimation model cannot accurately track the patient poses, so we trained our own custom pose estimation model with manual annotations for epileptic patients. The figure below shows the advantages of custom model, public pre-trained model cannot track all joints, and miss-track face keypoints.
<img src="https://github.com/xuyankun/STViG-for-READS-V/blob/main/compare.gif" width="500px">

You can test your own dataset (if your scenarios are very similar to ours, even patients with other movement-based diseases, e.g. PD) or fine-tune your own model with our provided custom pose estimation model [pose.pth](https://github.com/xuyankun/STViG-for-READS-V/blob/main/pose.pth). We utilize Openpose-lightweight to train the model, you can figure out how to implement it on your dataset in the original [Openpose-lightweight](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) project homepage. 

In this work, we propose a skeleton-based STViG model with a partitioning strategy to recognize seizure-related actions/behaviors, and generate probabilities rather than naive classification labels for timely detection. The figure below shows the performance of two seizures, $P$ means real-time predictive probabilities, and we set $5s$ video clip as target sample for analysis.
<img src="https://github.com/xuyankun/STViG-for-READS-V/blob/main/performance.gif" width="500px">



