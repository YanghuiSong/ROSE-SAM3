import unittest

from core.dag_residual_search import select_consensus


def fold_stat(unique_images=0, unique_pixels=0, violations=0):
    return {
        "unique_images": unique_images,
        "unique_pixels": unique_pixels,
        "anchor_violations": violations,
    }


class AnchorProtectionDAGTests(unittest.TestCase):
    def test_consensus_requires_same_fold_winner_in_four_of_five(self):
        candidates = [
            {"id": "c0:d1:good", "class_index": 0, "phrase": "good", "depth": 1},
            {"id": "c1:d1:stable", "class_index": 1, "phrase": "stable", "depth": 1},
            {"id": "c1:d1:unstable", "class_index": 1, "phrase": "unstable", "depth": 1},
        ]
        stats = {
            "c0:d1:good": [fold_stat(2, 20) for _ in range(5)],
            "c1:d1:stable": [fold_stat(2, 20) for _ in range(4)] + [fold_stat()],
            "c1:d1:unstable": [fold_stat(1, 10) for _ in range(3)] + [fold_stat()],
        }

        decisions, consensus = select_consensus(candidates, stats, class_count=2)

        self.assertEqual(len(decisions), 5)
        self.assertEqual(consensus, {"0": "c0:d1:good", "1": "c1:d1:stable"})

    def test_anchor_violation_makes_a_candidate_ineligible_in_that_fold(self):
        candidates = [
            {"id": "c0:d1:unsafe", "class_index": 0, "phrase": "unsafe", "depth": 1},
        ]
        stats = {
            "c0:d1:unsafe": [fold_stat(4, 50, violations=1)] + [fold_stat() for _ in range(4)],
        }

        decisions, consensus = select_consensus(candidates, stats, class_count=1)

        self.assertEqual(decisions, [{}, {}, {}, {}, {}])
        self.assertEqual(consensus, {})


if __name__ == "__main__":
    unittest.main()
