# DSG
# Dual-stage face generation for heterogeneous face recognition

## Requirements
1.Download pretrained model:
- [SimSwap](https://github.com/neuralchen/SimSwap)
  then unzip and put it to the folder ./pretrained_model/simswap.

2.Download ArcFace model from the above website. Then put it to the folder ./pretrained_model.

3.Download pretrained model:
- [Semi-HFR](https://github.com/xiyin11/Semi-HFR)
  and put it to the folder ./pretrained_model.

4.Download Tufts face dataset and unzip it. Then, put the visible face images to the folder ./tufts_dataset/vis and thermal face images to ./tufts_dataset/inf.

5.Download CelebA dataset and unzip it. Then, put all images to the folder ./celebA_dataset/data.

## Train
Before training or generation, please verify the correctness of all file paths referenced within the program.

1.Train Swapping Net

Run train_face_swapping.py

2.Train V2tT Net

Run train_v2tt.py

## Generation
Run generation.py. The results will be saved in the folder ./generated_images.

You can use the model trained by yourself or provided model.

## Model Download
The trained model files can be downloaded via the following links:
- [net1_40.pt（OneDrive）](https://1drv.ms/u/c/2c0f1036b31b3ed6/Ec9c44nYZDlIhFOkBD5BEA4BnIepfvpF1D2aLEzpUgU_ZQ?e=Razx71)
- After downloading, put it to the folder ./trained_model/provided

## more information please see the readme.pdf
 
