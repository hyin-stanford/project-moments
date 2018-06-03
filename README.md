# CS231n-project

You need to download FFmpeg

## Preparation


### Moments_in_Time_Mini 


I did not change the file for editing jpg and nframes. Actually, the code for kinetics works fine for Moments.

Convert 
```bash
python utils/video_jpg_kinetics.py avi_video_directory jpg_video_directory

For example:
python video_jpg_kinetics.py ../data/Moments_in_Time_Mini/training ../data/Moments_in_Time_Mini/jpg/training/
python video_jpg_kinetics.py ../data/Moments_in_Time_Mini/validation ../data/Moments_in_Time_Mini/jpg/validation/
```

* Generate n_frames files using ```utils/n_frames_kinetics.py```

```bash
python utils/n_frames_kinetics.py jpg_video_directory

For example:
python n_frames_kinetics.py ../data/Moments_in_Time_Mini/jpg/training
python n_frames_kinetics.py ../data/Moments_in_Time_Mini/jpg/validation
```

* Generate annotation file in json format similar to ActivityNet using ```utils/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
I have uploaded moments.json. Please download it.

Otherwise, you could also take a look at utils/minimoments_json.py
```


## Running the code

Assume the structure of data directories is the following:

```misc
~/
  data/
    Moments_in_Time_Mini/
      jpg/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    results/
      .../ (directories of model names, where we store history of different models)
      save_100.pth
    moments.json
```

Confirm all options.

```bash
python main.lua -h
```

Train ResNets-34 on the Kinetics dataset (400 classes) with 4 CPU threads (for data loading).  
Batch size is 128.  
Save models at every 5 epochs.
All GPUs is used for the training.
If you want a part of GPUs, use ```CUDA_VISIBLE_DEVICES=...```.

```bash
python main.py --root_path ./data --video_path Moments_in_Time_Mini/jpg --annotation_path moments.json 
--result_path results --dataset moments --model resnet --model_depth 10 --n_classes 200 --batch_size 128 --n_threads 4 --checkpoint 5
```

## Use the pretrained parameters.

Put the pretrained parameters downloadable at https://drive.google.com/open?id=14KRBqT8ySfPtFSuLsFS2U4I-ihTDs0Y9 at '~/data/models/resnet-34-kinetics.pth', and run code

```bash
python main.py --root_path ./data --video_path Moments_in_Time_Mini/jpg --annotation_path moments.json --result_path results --dataset moments --n_classes 400 --n_finetune_classes 200 --pretrain_path models/resnet-101-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 101 --resnet_shortcut B --batch_size 30 --n_threads 4 --checkpoint 5
```
```bash
python main.py --root_path ./data --video_path Moments_in_Time_Mini/jpg --annotation_path moments.json --result_path results 
--dataset moments --n_classes 400 --n_finetune_classes 200 --pretrain_path models/resnet-34-kinetics.pth 
--ft_begin_index 4 --model resnet --model_depth 34 --resnet_shortcut A --batch_size 30 --n_threads 4 --checkpoint 5
```
The parameters used for other models are listed here:
```bash
resnet-18-kinetics.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet-34-kinetics.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet-34-kinetics-cpu.pth: CPU ver. of resnet-34-kinetics.pth
resnet-50-kinetics.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet-101-kinetics.pth: --model resnet --model_depth 101 --resnet_shortcut B
resnet-152-kinetics.pth: --model resnet --model_depth 152 --resnet_shortcut B
resnet-200-kinetics.pth: --model resnet --model_depth 200 --resnet_shortcut B
preresnet-200-kinetics.pth: --model preresnet --model_depth 200 --resnet_shortcut B
wideresnet-50-kinetics.pth: --model wideresnet --model_depth 50 --resnet_shortcut B --wide_resnet_k 2
resnext-101-kinetics.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
densenet-121-kinetics.pth: --model densenet --model_depth 121
densenet-201-kinetics.pth: --model densenet --model_depth 201
```
## Plot train history

If you train a new model, the training and validate history of the old model is going to get erased. So you want to store your old parameter at '~/data/results/model_name(resnet34, for example)/'


```bash
python plothistory --model resnet34
```

## Continue the training process.

Since I only train on pretrained model by now, see the latter part for reading pretrained parameters and used trained parameters.
```bash
python main.py --root_path ./data --video_path Moments_in_Time_Mini/jpg --annotation_path moments.json --result_path results --dataset moments 
--resume_path results/save_5.pth --model_depth 34 --n_classes 200 --batch_size 30 --n_threads 4 --checkpoint 5
```

## Continue the training process from pretrained and old trained parameters.
If you train based on pretrained model, then you should use this code for resume training.
```bash
python main.py --root_path ./data --video_path Moments_in_Time_Mini/jpg --annotation_path moments.json --result_path results --dataset moments --n_classes 400 --n_finetune_classes 200 
--pretrain_path models/resnet-34-kinetics.pth --resume_path results/save_15.pth --ft_begin_index 4 --model resnet --model_depth 34 --resnet_shortcut A --batch_size 30 --n_threads 4 --checkpoint 5
```


## Print out some wrongly classified samples.

This part is for an intuitive understanding of wrongly classified samples.
```bash
python main.py ---root_path ./data --video_path Moments_in_Time_Mini/jpg --annotation_path moments.json --result_path results --dataset moments --n_classes 400 --n_finetune_classes 200 
--pretrain_path models/resnet-34-kinetics.pth --resume_path results/save_15.pth --ft_begin_index 4 --debug --model resnet 
--model_depth 34 --resnet_shortcut A --batch_size 30 --n_threads 4 --checkpoint 5
```

