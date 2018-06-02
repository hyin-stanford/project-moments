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
python main.py --root_path ./data --video_path Moments_in_Time_Mini/jpg --annotation_path moments.json --result_path results 
--dataset moments --n_classes 400 --n_finetune_classes 200 --pretrain_path models/resnet-34-kinetics.pth 
--ft_begin_index 4 --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --n_threads 4 --checkpoint 5

```
## Plot train history

If you train a new model, the training and validate history of the old model is going to get erased. So you want to store your old parameter at '~/data/results/model_name(resnet34, for example)/'


```bash
python plothistory --model resnet34
```