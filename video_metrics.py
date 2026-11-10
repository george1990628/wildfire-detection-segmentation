"""Label-aware video scoring, maintained from the historical Pioneer notebook."""
import math

LABELS = ('Fire', 'No_Fire')


def labels_from_scores(scores, positive_class, threshold=0.5):
    """Map a binary sigmoid score to its declared training label."""
    if positive_class not in LABELS:
        raise ValueError('positive_class must be Fire or No_Fire')
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between zero and one')
    negative_class = next(label for label in LABELS if label != positive_class)
    predictions = []
    for value in scores:
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Each sigmoid score must be finite and in [0, 1]')
        predictions.append(positive_class if score >= threshold else negative_class)
    return predictions


def classification_metrics(truth, predictions):
    """Return observed frame metrics; never invent labels for an unlabeled video."""
    truth, predictions = list(truth), list(predictions)
    if not truth or len(truth) != len(predictions):
        raise ValueError('Nonempty, equally sized labels and predictions are required')
    if any(label not in LABELS for label in truth + predictions):
        raise ValueError('Labels must be Fire or No_Fire')
    matrix = [[sum(t == actual and p == predicted for t, p in zip(truth, predictions))
               for predicted in LABELS] for actual in LABELS]
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        fp = sum(row[i] for j, row in enumerate(matrix) if j != i)
        fn = sum(value for j, value in enumerate(matrix[i]) if j != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'support': sum(matrix[i]),
        }
    return {'n_frames': len(truth), 'accuracy': sum(matrix[i][i] for i in range(2)) / len(truth),
            'label_order': list(LABELS), 'confusion_matrix': matrix, 'per_class': per_class,
            'zero_division': 0}
