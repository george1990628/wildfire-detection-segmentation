# Changes to the original notebook

The scripts come from my original Pioneer notebook, with the following fixes and cleanup. These changes do not add new research results.

- Removed saved outputs and personal Drive paths. Setup no longer deletes `frames/Test/*`.
- Made the video class mapping and RGB/BGR order explicit. The FLAME training code documents Fire=0 and No_Fire=1, while the old video code treated a high score as Fire. Check the settings used to train each checkpoint.
- Removed the assumption that every video frame contains fire. The evaluator now calculates accuracy only when labels are provided.
- Matched images and masks by filename before shuffling. The old code shuffled the two lists separately. Both training and validation must have matching pairs, and the validation split must contain at least one pair.
- Kept nearest-neighbor resizing for masks and treated nonzero pixels as fire.
- Replaced unthresholded `MeanIoU` with `BinaryIoU` at a threshold of 0.5. The checkpoint callback now monitors `val_binary_iou`. Scores from this metric should not be treated as interchangeable with the old logs.
- Replaced the normalization Lambda with `Rescaling(1/255)` and saved new models as `.keras` files. Old checkpoints are unchanged.
- Added an image-decoding check and displayed predicted masks as two-dimensional arrays.

The split is still based on image/mask pairs. Similar frames from one video can appear in both splits, so a separate video- or event-level split would be needed to test generalization more carefully.

Metric reference: [TensorFlow BinaryIoU](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryIoU).
