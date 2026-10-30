import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import process_text, build_freqs, extract_features, FeatureScaler, gradient_descent, sigmoid, score, run


class TestSentimentPipeline(unittest.TestCase):
    def test_preprocessing_and_negation(self):
        tokens = process_text("I didn't enjoy the terrible movie!")
        self.assertIn('not', tokens)
        self.assertIn('terribl', tokens)
        self.assertNotIn('the', tokens)

    def test_word_frequencies_and_features(self):
        texts = ['good fun movie', 'bad dull movie']
        freq = build_freqs(texts, np.array([1,0]))
        self.assertEqual(freq[('good',1)],1)
        v = extract_features('good bad', freq)
        self.assertEqual(v.shape,(3,))
        self.assertEqual(v.tolist(), [1.0,1.0,1.0])

    def test_gradient_decent_improves_loss(self):
        X = np.array([[1,-1], [1,-.5], [1,.5], [1,1]], dtype=float)
        y = np.array([0,0,1,1])
        theta, history = gradient_descent(X, y, .1, 300)
        self.assertLess(history[-1][1], history[0][1])
        self.assertEqual(score(X,y,theta)['accuracy'],1.0)

    def test_scaler_preserves_bias(self):
        X=np.array([[1,0,2],[1,10,4]], dtype=float)
        scaled=FeatureScaler().fit(X).transform(X)
        np.testing.assert_array_equal(scaled[:,0],[1,1])

    def test_offline_smoke_end_to_end(self):
        # Лише технічний мінікорпус для тесту — НЕ оцінка на даних IMDb.
        positives = [f'This fantastic good excellent movie is lovely and wonderful number {i}' for i in range(30)]
        negatives = [f'This horrible bad awful movie is boring and terrible number {i}' for i in range(30)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'toy.txt'
            with path.open('w',encoding='utf-8') as out:
                for text in positives: out.write(f'{text}\t1\n')
                for text in negatives: out.write(f'{text}\t0\n')
            result=run(path,Path(directory)/'results')
            self.assertGreaterEqual(result['accuracy'],0.8)
            self.assertTrue((Path(directory)/'results'/'metrics.json').exists())
            self.assertTrue((Path(directory)/'results'/'training_loss.png').exists())


if __name__ == '__main__':
    unittest.main()
