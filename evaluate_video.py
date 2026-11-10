"""Evaluate a supplied binary fire checkpoint without assuming ground truth.

Maintained from the uploaded 2025 Pioneer notebook. Color order and score
mapping must agree with the checkpoint's actual training configuration.
"""
import argparse
import csv
import json
from pathlib import Path

from video_metrics import LABELS, classification_metrics, labels_from_scores


def read_frame_labels(path, count):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['frame_index', 'label']:
            raise ValueError('Label CSV must have columns frame_index,label')
        indexed = {}
        for row in reader:
            index = int(row['frame_index'])
            if index in indexed or row['label'] not in LABELS:
                raise ValueError('Duplicate frame index or unsupported label')
            indexed[index] = row['label']
    if set(indexed) != set(range(count)):
        raise ValueError('Frame labels must cover every decoded frame exactly once')
    return [indexed[i] for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--positive-class', choices=LABELS, required=True,
                        help='Class encoded as 1 during training; high sigmoid score')
    parser.add_argument('--color-order', choices=['RGB', 'BGR'], required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--size', type=int, default=256, help='Historical classifier input size')
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument('--video-label', choices=LABELS,
                        help='Explicitly assert a uniform label for every decoded frame')
    labels.add_argument('--frame-labels', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/video_evaluation.json'))
    args = parser.parse_args()
    if args.size <= 0 or not 0 <= args.threshold <= 1:
        parser.error('size must be positive and threshold in [0, 1]')
    if not args.model.is_file() or not args.video.is_file():
        parser.error('model and video must be existing local files')

    import cv2
    import numpy as np
    from tensorflow.keras.models import load_model

    model = load_model(args.model, compile=False)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise ValueError('Cannot open video')
    scores = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.color_order == 'RGB':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (args.size, args.size)).astype(np.float32)
            prediction = np.asarray(model.predict(frame[None, ...], verbose=0))
            if prediction.size != 1:
                raise ValueError('Expected one binary sigmoid score per frame')
            scores.append(float(prediction.reshape(-1)[0]))
    finally:
        capture.release()
    if not scores:
        raise ValueError('No frames were decoded; no metrics were calculated')

    predictions = labels_from_scores(scores, args.positive_class, args.threshold)
    result = {
        'model': str(args.model), 'video': str(args.video),
        'positive_class': args.positive_class, 'color_order': args.color_order,
        'threshold': args.threshold, 'size': args.size, 'n_frames': len(scores),
        'prediction_counts': {label: predictions.count(label) for label in LABELS},
        'mean_sigmoid_score': sum(scores) / len(scores),
        'ground_truth_source': None, 'metrics': None,
        'scores': scores, 'predictions': predictions,
    }
    if args.video_label is not None:
        truth = [args.video_label] * len(scores)
        result['ground_truth_source'] = {'uniform_video_label': args.video_label}
        result['metrics'] = classification_metrics(truth, predictions)
    elif args.frame_labels is not None:
        truth = read_frame_labels(args.frame_labels, len(scores))
        result['ground_truth_source'] = {'frame_label_csv': str(args.frame_labels)}
        result['metrics'] = classification_metrics(truth, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'scores', 'predictions'}}, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
