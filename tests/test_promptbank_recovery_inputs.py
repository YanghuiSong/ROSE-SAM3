import json
import tempfile
import unittest
from pathlib import Path

from core.promptbank_compiler import (
    build_recovery_inputs, list_calibration_images, stage1_calibration,
    validate_residual_calibration,
)


class PromptBankRecoveryInputTests(unittest.TestCase):
    def test_residual_manifest_records_the_exact_four_dataset_cohorts(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo) as temporary_directory:
            train = Path(temporary_directory)
            for dataset, count in (("openearthmap", 512), ("udd5", 120), ("vdd", 280), ("vaihingen", 344)):
                with self.subTest(dataset=dataset):
                    source = repo / "calibrated_prompts/recovery_inputs/residual_calibration" / f"{dataset}.json"
                    manifest = json.loads(source.read_text())
                    images = [train / relative for relative in manifest["relative_images"]]
                    self.assertEqual(len(images), count)
                    self.assertEqual(
                        validate_residual_calibration(repo, dataset, train, images, manifest["partitions"]),
                        source,
                    )

    def test_equal_image_counts_do_not_hide_a_different_residual_cohort(self):
        repo = Path(__file__).resolve().parents[1]
        source = repo / "calibrated_prompts/recovery_inputs/residual_calibration/udd5.json"
        manifest = json.loads(source.read_text())
        with tempfile.TemporaryDirectory(dir=repo) as temporary_directory:
            train = Path(temporary_directory)
            images = [train / relative for relative in manifest["relative_images"]]
            images[0] = train / "different_training_image.JPG"
            with self.assertRaisesRegex(ValueError, "cohort/folds differ"):
                validate_residual_calibration(repo, "udd5", train, images, manifest["partitions"])

    def test_oem_historical_base_calibration_is_explicitly_val_not_train(self):
        repo = Path(__file__).resolve().parents[1]
        manifest = json.loads((repo / "calibrated_prompts/recovery_inputs/base_calibration/openearthmap.json").read_text())
        with tempfile.TemporaryDirectory(dir=repo) as temporary_directory:
            root = Path(temporary_directory)
            train = root / "img_dir/train"
            val = root / "img_dir/val"
            train.mkdir(parents=True)
            for relative in manifest["relative_images"]:
                path = val / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            images, partitions, split, source = stage1_calibration(repo, "openearthmap", train)
            self.assertEqual(split, "val")
            self.assertEqual(len(images), 192)
            self.assertTrue(all(path.is_relative_to(val) for path in images))
            self.assertEqual(sorted(index for ids in partitions.values() for index in ids), list(range(192)))

    def test_missing_historical_images_do_not_silently_use_other_train_images(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo) as temporary_directory:
            train = Path(temporary_directory) / "train"
            train.mkdir()
            (train / "unrelated.png").touch()
            with self.assertRaises(FileNotFoundError):
                stage1_calibration(repo, "udd5", train)

    def test_stage1_uses_the_historical_finite_dag_vocabulary(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo) as temporary_directory:
            inputs = build_recovery_inputs(repo, "udd5", Path(temporary_directory))
            queries = inputs["query_bank"].read_text(encoding="utf-8")
            phrases = {phrase for row in queries.splitlines() for phrase in row.split(",")}
            self.assertEqual(len(phrases), 20)
            self.assertNotIn("herbaceous vegetation", phrases)
            self.assertIn("plant cover", phrases)

    def test_vaihingen_proposal_can_be_scored_against_ov3_car_anchor(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo) as temporary_directory:
            inputs = build_recovery_inputs(
                repo,
                "vaihingen",
                Path(temporary_directory),
            )
            native = json.loads(inputs["classes"].read_text(encoding="utf-8"))
            proposals = json.loads(inputs["base_bank_candidates"].read_text(encoding="utf-8"))
            dag = json.loads(inputs["candidate_dag"].read_text(encoding="utf-8"))

            self.assertEqual(native["classes"][4], "car")
            self.assertEqual(proposals["banks"][0][4][0], "small vehicle")
            self.assertTrue(any(
                node["immutable"]
                and node["class_index"] == 4
                and node["phrase"] == "car"
                for node in dag["nodes"]
            ))

    def test_sampling_uses_all_small_train_splits_and_caps_at_512(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo) as temporary_directory:
            root = Path(temporary_directory)
            oem = root / "oem_train"
            udd5 = root / "udd5_train"
            oem.mkdir()
            udd5.mkdir()
            for index in range(520):
                (oem / f"tile_{index:04d}.png").touch()
            for index in range(120):
                (udd5 / f"tile_{index:04d}.png").touch()

            selected_oem = list_calibration_images(oem, "openearthmap")
            selected_oem_again = list_calibration_images(oem, "openearthmap")
            selected_udd5 = list_calibration_images(udd5, "udd5")

            self.assertEqual(len(selected_oem), 512)
            self.assertEqual(selected_oem, selected_oem_again)
            self.assertEqual(len(selected_udd5), 120)
            self.assertEqual(selected_udd5, sorted(udd5.glob("*.png")))


if __name__ == "__main__":
    unittest.main()
