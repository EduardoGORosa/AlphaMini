# test_mcts.py

import unittest
from mcts import MCTS
from chess_env import ChessEnv
import torch
import numpy as np
import torch.nn as nn

class MockModel(nn.Module):
    def __init__(self, policy_probs, value_out):
        super().__init__()
        self.policy_probs = policy_probs  # numpy array [20480]
        self.value_out = value_out        # scalar
        self.dummy = nn.Parameter(torch.zeros(1))  # Dummy parameter to avoid StopIteration

    def forward(self, x):
        batch_size = x.size(0)
        policy = torch.tensor([self.policy_probs for _ in range(batch_size)], dtype=torch.float32)
        value = torch.tensor([[self.value_out]] * batch_size, dtype=torch.float32)
        return policy, value

class TestMCTSReRooting(unittest.TestCase):
    def setUp(self):
        # Initialize a MockModel that assigns high probability to specific Black moves
        policy_probs = np.ones(20480, dtype=np.float32) * 1e-6  # Initialize with near-zero probabilities

        # Assign high probabilities to specific Black moves after 'e2e4'
        # Assuming 'e2e4' index=0, 'd2d4'=1, 'c2c3'=2 (White moves)
        # Black moves: 'c7c5'=3, 'e7e5'=4, 'd7d5'=5
        policy_probs[3] = 0.6  # 'c7c5'
        policy_probs[4] = 0.3  # 'e7e5'
        policy_probs[5] = 0.1  # 'd7d5'

        # Normalize the probabilities
        policy_probs /= policy_probs.sum()

        value_out = 0.0  # Neutral value

        self.model = MockModel(policy_probs, value_out)
        self.model.eval()

        # Define move_to_index including both White and Black moves
        self.move_to_index = {
            'e2e4': 0,
            'd2d4': 1,
            'c2c3': 2,
            'c7c5': 3,
            'e7e5': 4,
            'd7d5': 5
        }

        # Initialize MCTS with the MockModel
        self.mcts = MCTS(self.model, self.move_to_index)

        # Create a ChessEnv and make the initial move 'e2e4'
        self.env = ChessEnv()
        self.env.step('e2e4')  # White's move

        # Initialize the MCTS root with the current environment
        self.mcts.initialize_root(self.env)

        # Run simulations to expand Black's possible moves
        self.mcts.run_simulations(num_simulations=6, batch_size=2)  # Adjust as needed

    def test_re_root_existing_move(self):
        # Re-root to 'c7c5', which should be a child with high visit counts
        move = 'c7c5'

        # Ensure 'c7c5' is a child of the root
        self.assertIn(move, self.mcts.root.children)

        # Re-root to 'c7c5'
        self.mcts.re_root(move)

        # Verify that the new root corresponds to 'c7c5'
        last_move = self.mcts.root.env.get_last_move()
        self.assertEqual(last_move, move)

    def test_re_root_new_move(self):
        # Re-root to 'e7e5', another valid Black move
        move = 'e7e5'

        # Ensure 'e7e5' is a child of the root
        self.assertIn(move, self.mcts.root.children)

        # Re-root to 'e7e5'
        self.mcts.re_root(move)

        # Verify that the new root corresponds to 'e7e5'
        last_move = self.mcts.root.env.get_last_move()
        self.assertEqual(last_move, move)

    def test_visit_counts_after_re_rooting(self):
        # Run additional simulations before re-rooting
        self.mcts.run_simulations(num_simulations=4, batch_size=2)

        move = 'c7c5'

        # Fetch visit count before re-rooting
        child_node = self.mcts.root.children.get(move, None)
        self.assertIsNotNone(child_node, "Move 'c7c5' should exist with a visit count.")
        visit_before = child_node.visit_count

        # Re-root to 'c7c5'
        self.mcts.re_root(move)

        # New root should have the same visit count
        new_root = self.mcts.root
        self.assertEqual(new_root.visit_count, visit_before)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
