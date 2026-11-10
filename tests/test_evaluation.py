import csv
from pathlib import Path
import tempfile
import unittest

from dataset_pairs import split_pairs
from evaluate_video import read_frame_labels
from video_metrics import classification_metrics, labels_from_scores


class EvaluationChecks(unittest.TestCase):
    def test_training_label_mapping_can_be_either_direction(self):
        self.assertEqual(labels_from_scores([0.1, 0.9], 'No_Fire'), ['Fire', 'No_Fire'])
        self.assertEqual(labels_from_scores([0.1, 0.9], 'Fire'), ['No_Fire', 'Fire'])

    def test_one_predicted_class_is_not_assumed_perfect(self):
        result = classification_metrics(['Fire', 'No_Fire'], ['No_Fire', 'No_Fire'])
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['confusion_matrix'], [[0, 1], [0, 1]])
        self.assertEqual(result['per_class']['Fire']['recall'], 0)

    def test_unlabeled_or_mismatched_input_is_rejected(self):
        for truth, predictions in [([], []), (['Fire'], ['Fire', 'No_Fire'])]:
            with self.assertRaises(ValueError):
                classification_metrics(truth, predictions)

    def test_invalid_scores_are_rejected(self):
        for value in [float('nan'), float('inf'), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                labels_from_scores([value], 'Fire')

    def test_sparse_frame_annotations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.csv'
            path.write_text('frame_index,label\n0,Fire\n')
            self.assertEqual(read_frame_labels(path, 1), ['Fire'])
            with self.assertRaises(ValueError):
                read_frame_labels(path, 2)

    def test_pairs_stay_matched_and_validation_is_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            images, masks = Path(directory)/'images', Path(directory)/'masks'
            images.mkdir(); masks.mkdir()
            for name in ['a', 'b']:
                (images/(name+'.jpg')).touch()
                (masks/(name+'.png')).touch()
            training, validation = split_pairs(images, masks)
            self.assertEqual((len(training), len(validation)), (1, 1))
            for image, mask in training + validation:
                self.assertEqual(Path(image).stem, Path(mask).stem)
            (masks/'b.png').unlink()
            with self.assertRaises(ValueError):
                split_pairs(images, masks)


if __name__ == '__main__':
    unittest.main()
