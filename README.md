# Wildfire detection and segmentation

For my Pioneer Academics research, I worked on classifying wildfire images and using U-Net to identify the parts of an image that contain fire. I also wrote code to test a classifier on video frames and view segmentation predictions.

Classifier training uses Alireza Shamsoshoara and collaborators' [FLAME repository](https://github.com/AlirezaShamsoshoara/Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle). This repo contains my experiment setup, video analysis, segmentation code, and frame-extraction tool.

## Files

| File | What it does |
| --- | --- |
| [Project notebook](notebooks/pioneer_workflow.ipynb) | Explains the setup and how to use the scripts. |
| [segmentation_training.py](segmentation_training.py) | Trains a U-Net on 512×512 RGB images and binary masks. |
| [segmentation_evaluation.py](segmentation_evaluation.py) | Displays a predicted fire mask for an image. |
| [evaluate_video.py](evaluate_video.py) | Runs a classifier on each video frame and calculates metrics when labels are provided. |
| [frame_extractor.py](frame_extractor.py) | Saves a selected frame from a video. |

## Setup

```bash
python -m pip install -r requirements.txt
```

For classifier training, follow the FLAME repository's data and configuration instructions. Keep a record of the code version, class order, input size, and preprocessing used for each model. Older checkpoints may need their original TensorFlow/Keras environment.

For segmentation, place JPG images in `frames/Segmentation/Data/Images` and PNG masks in `frames/Segmentation/Data/Masks`. Each image and mask must have the same filename before the extension. Then run:

```bash
python segmentation_training.py
```

The U-Net uses an 85/15 training/validation split with seed 1337, 30 epochs, and a batch size of 16. Nonzero mask pixels are treated as fire. Frames from the same video can be very similar, so this split alone does not show how well the model works on a new location or fire event.

## Test a video

Set the class that a high sigmoid score represents and the color order used during training. The FLAME training code documents Fire=0, No_Fire=1 with RGB input. For a checkpoint trained with those settings:

```bash
python evaluate_video.py --model /path/to/model.h5 --video /path/to/video.mov --positive-class No_Fire --color-order RGB
```

Check the training setup before using those flags with a different checkpoint.

Without labels, the script reports predictions and scores. To calculate accuracy, use `--video-label Fire` only if every frame contains fire, or provide `--frame-labels labels.csv` with a `frame_index,label` row for every decoded frame. Nearby frames are correlated, so frame-level scores should be interpreted carefully.

## Notes

The original video code assumed every frame contained fire and used an uncertain score-to-class mapping. Those issues were corrected, so I have not carried over its old accuracy numbers. The segmentation example displays a prediction rather than calculating a test-set IoU.

See [the code changes](docs/maintenance.md) and [references](THIRD_PARTY_NOTICES.md). This repo includes code only.
