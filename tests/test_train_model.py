import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_model import get_feature_columns, load_training_data, resolve_data_path


class TrainModelTests(unittest.TestCase):
    def test_uses_current_car_data_dataset(self):
        data_path = resolve_data_path()

        self.assertTrue(os.path.exists(data_path))
        self.assertEqual(os.path.basename(data_path), "car_data.csv")

        df = load_training_data(data_path)
        self.assertIn("Selling_Price", df.columns)
        self.assertTrue(set(get_feature_columns()).issubset(df.columns))


if __name__ == "__main__":
    unittest.main()
