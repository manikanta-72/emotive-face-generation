# Emotive-face-generation
A course project using Deep Generative Models

## Dataset Download
We used Edges2Shoes, Night2Day, CelebA, FERDB datasets for this project. <br />
Edges2Shoes, Night2Day datasets can be downloaded by running the following bash script <br />
``` download_pix2pix_dataset.sh <dataset> ```
Other datasets can be found [here](https://drive.google.com/drive/folders/1kkCmYKXRQHTn7lJ_FbBZZPHgEGV48QAs?usp=share_link)

## Pre trained models
Pre trained models for each experiment can be found here

## Training model
From within model folder run
```
python train.py --dataroot ../datasets/night2day/ --name augcgan_model_night2day
```

## Testing & Evaluation
From within model folder run (example for night2day dataset)
```
python .\infer.py --chk_path .\checkpoints\augcgan_model\latest --res_dir res --datarootA ..\datasets\night2day\valA\ --datarootB ..\datasets\night2day\valB\ --metric visual
```
