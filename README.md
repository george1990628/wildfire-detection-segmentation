# Wildfire detection and segmentation

**Status: historical Pioneer Academics research (2025), with disclosed maintenance for public sharing.**

Aerial fire-image classification, video-frame analysis, and U-Net segmentation experiments. The original notebook uses Alireza Shamsoshoara and collaborators' FLAME repository for classifier training; the uploaded code adds custom video analysis, a U-Net training workflow, custom-image prediction, and frame extraction.

## Repository contents

| File | Purpose |
| --- | --- |
| [Pioneer workflow notebook](notebooks/pioneer_workflow.ipynb) | Data/setup instructions and links to the maintained scripts. |
| [segmentation_training.py](segmentation_training.py) | 512×512 RGB U-Net, paired image/mask split, 30 epochs, batch size 16. |
| [segmentation_evaluation.py](segmentation_evaluation.py) | Visualize a predicted binary mask on a supplied image. |
| [evaluate_video.py](evaluate_video.py) | Frame-by-frame prediction with explicit class mapping/color order and optional supplied labels. |
| [frame_extractor.py](frame_extractor.py) | Save a selected frame from a video. |

## Setup

```bash
python -m pip install -r requirements.txt
```

The original notebook did not preserve an exact environment. TensorFlow/Keras compatibility must be checked for historical checkpoints; no old checkpoint is converted or bundled here.

For classifier training, use the [upstream FLAME repository](https://github.com/AlirezaShamsoshoara/Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle), obtain its separately hosted dataset, and follow its configuration. The upstream classifier is not repackaged as an original model here. Record the repository commit, saved class names, input dimensions, and preprocessing when running new experiments.

For segmentation, supply matching `*.jpg` images and `*.png` masks with identical filename stems under `frames/Segmentation/Data/Images` and `frames/Segmentation/Data/Masks`, then run:

```bash
python segmentation_training.py
```

The implementation preserves the historical 85/15 pair split using seed 1337 and the U-Net channel layout. Binary masks use nonzero = fire. A frame-level random split may leak near-duplicate frames from the same source video; it is not evidence of geographic or event-level generalization.

## Video prediction

The evaluator requires the class represented by a high sigmoid score and the color order expected by the checkpoint. The current upstream training source documents Fire=0, No_Fire=1 and uses RGB input; a different historical checkpoint must be checked against its own training setup.

For a checkpoint confirmed to follow that setup:

```bash
python evaluate_video.py --model /path/to/model.h5 --video /path/to/video.mov --positive-class No_Fire --color-order RGB
```

Without labels, the script reports prediction counts and scores, not accuracy. Optional `--video-label Fire` explicitly asserts that every decoded frame has that label. For varying labels, provide `--frame-labels labels.csv`, with one `frame_index,label` entry for every decoded frame. Adjacent-frame results do not represent independent test samples.

## Results and limitations

Historical results are not re-established during portfolio preparation. No numerical accuracy claim is carried over from the original video evaluator, which assumed all frames were fire and had an ambiguous score-to-class mapping. The custom-image segmentation example is a visualization. See [maintenance notes](docs/maintenance.md) for corrections and [attribution](THIRD_PARTY_NOTICES.md).

The public package includes code only.
