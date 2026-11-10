"""Unit tests for landing outcomes without running the simulator."""

from types import SimpleNamespace
import unittest

import landing_stats


class ClassifyOutcomeTests(unittest.TestCase):
    def test_crashed(self):
        game = SimpleNamespace(game_over=True, lander=SimpleNamespace(awake=True))

        self.assertEqual(landing_stats.classify_outcome(game, [0.0]), "crashed")

    def test_crashed_takes_precedence_over_flying_away(self):
        game = SimpleNamespace(game_over=True, lander=SimpleNamespace(awake=False))

        for x in (-1.1, 1.1):
            with self.subTest(x=x):
                self.assertEqual(landing_stats.classify_outcome(game, [x]), "crashed")

    def test_flew_away_on_both_sides_including_screen_edges(self):
        game = SimpleNamespace(game_over=False, lander=SimpleNamespace(awake=False))

        for x in (-1.1, -1.0, 1.0, 1.1):
            with self.subTest(x=x):
                self.assertEqual(landing_stats.classify_outcome(game, [x]), "flew_away")

    def test_landed_including_pad_edges(self):
        game = SimpleNamespace(game_over=False, lander=SimpleNamespace(awake=False))

        for x in (-landing_stats.PAD_HALF_WIDTH, 0.0, landing_stats.PAD_HALF_WIDTH):
            with self.subTest(x=x):
                self.assertEqual(landing_stats.classify_outcome(game, [x]), "landed")

    def test_landed_just_outside_pad(self):
        game = SimpleNamespace(game_over=False, lander=SimpleNamespace(awake=False))
        outside_pad = landing_stats.PAD_HALF_WIDTH + 0.001

        for x in (-outside_pad, outside_pad):
            with self.subTest(x=x):
                self.assertEqual(landing_stats.classify_outcome(game, [x]), "landed_off_pad")

    def test_never_settled_while_awake_on_screen(self):
        game = SimpleNamespace(game_over=False, lander=SimpleNamespace(awake=True))

        for x in (-0.99, 0.0, 0.99):
            with self.subTest(x=x):
                self.assertEqual(landing_stats.classify_outcome(game, [x]), "never_settled")


if __name__ == "__main__":
    unittest.main()
