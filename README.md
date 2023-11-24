# STViG-for-READS-V
The code will be released soon!
In this project, we propose a novel skeleton-based spatiotemporal vision graph neural network (STViG) for efficient, accurate, and timely REal-time Automated Detection of epileptic Seizures from surveillance Videos (READS-V).
Previous video-based seizure detection studies did not rely on skeleton information because the public pre-tained pose estimation model cannot track the patient poses accurately. We manually annotate the pose label for the patients, and train our own custom for patient pose estimation. 

You can test your own dataset (patient dataset) or fine-tune your own model with our provided custom pose estimation model [pose.pth](https://github.com/xuyankun/STViG-for-READS-V/blob/main/pose.pth). We utilize Openpose-lightweight to train the model, you can figure out how to implement it in the original [Openpose-lightweight](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) project homepage.
