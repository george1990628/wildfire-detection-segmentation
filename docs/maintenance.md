# Public maintenance notes

The supplied notebook is the source of the maintained scripts. This packaging step did not generate new research results.

- Saved plots and outputs and personal Drive paths are removed. Setup no longer deletes `frames/Test/*`.
- Classification training remains attributed to the upstream FLAME repository. Its current source documents Fire=0 and No_Fire=1. The historical uploaded video code interpreted a high score as Fire and always invented Fire ground truth. The public evaluator requires explicit score mapping and RGB/BGR choice, and calculates metrics only from user-supplied labels.
- The segmentation split now joins images and masks by identical filename stem before shuffling, rather than independently shuffling two lists. Datasets whose names follow another convention must be explicitly mapped first.
- A nonempty validation split and matching image/mask sets are required. This is still a pair-level split; video/event leakage remains a limitation.
- Mask resizing explicitly uses nearest-neighbor interpolation and the original nonzero foreground convention.
- `BinaryIoU` thresholds sigmoid predictions at 0.5 and the checkpoint callback monitors that named metric. This replaces the original unthresholded `MeanIoU` configuration. New scores are not directly interchangeable with historical logs.
- The input-normalization Lambda is replaced by the equivalent `Rescaling(1/255)` layer for new-model serialization. Newly trained checkpoints use `.keras`; old model files are not rewritten.
- Custom-image prediction checks decoding and displays a two-dimensional binary mask.

Reference for the metric: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryIoU
