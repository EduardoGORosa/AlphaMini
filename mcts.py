import math
import numpy as np
from collections import defaultdict
import torch
from utils import encode_state

class MCTSNode:
    """
    Represents a node in the MCTS tree.
    """
    def __init__(self, state, parent=None):
        self.state = state  # ChessEnv instance (cloned state)
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0

    def is_fully_expanded(self):
        """Check if all legal moves have been expanded."""
        return len(self.children) == len(self.state.get_legal_moves())

    def value(self):
        """Return the average value of this node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    """
    Implements the Monte Carlo Tree Search algorithm.
    """
    def __init__(self, model, move_to_index, index_to_move, c_puct=1.4):
        """
        Args:
            model: Neural network for policy and value evaluation.
            move_to_index: Dict mapping UCI moves to indices.
            index_to_move: Dict mapping indices back to UCI moves.
            c_puct: Exploration constant.
        """
        self.model = model
        self.move_to_index = move_to_index
        self.index_to_move = index_to_move
        self.c_puct = c_puct  # Exploration constant
        self.Q = defaultdict(float)  # Action value
        self.N = defaultdict(int)    # Visit count
        self.P = defaultdict(float)  # Prior probabilities

    def search(self, env, simulations=100):
        """
        Perform MCTS simulations and return the best move as a UCI string.
        """
        root = MCTSNode(env.clone())

        for _ in range(simulations):
            node = root
            path = []

            # Selection
            while not node.state.is_game_over() and node.is_fully_expanded():
                move, node = self.select_child(node)
                path.append((node, move))

            # Expansion and Evaluation
            if not node.state.is_game_over():
                legal_moves = node.state.get_legal_moves()
                for move in legal_moves:
                    if move not in node.children:
                        child_state = node.state.clone()
                        child_state.step(move)
                        node.children[move] = MCTSNode(child_state, parent=node)

                # Neural network evaluation
                state_tensor = encode_state(node.state.get_state_representation())
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                state_tensor = state_tensor.to(next(self.model.parameters()).device)

                with torch.no_grad():
                    policy_logits, value = self.model(state_tensor)
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                value = value.item()

                for move in legal_moves:
                    if move in self.move_to_index:
                        idx = self.move_to_index[move]
                        self.P[move] = policy[idx] if policy[idx] > 0 else 1e-6

            # Backpropagation
            for (node, move) in path:
                self.N[move] += 1
                self.Q[move] += (value - self.Q[move]) / self.N[move]

        # Choose the move with the highest visit count
        visit_counts = {move: self.N[move] for move in root.children}
        best_move = max(visit_counts, key=visit_counts.get)
        return best_move  # Return the best move as a UCI string

    def select_child(self, node):
        """
        Select the child node with the highest Upper Confidence Bound (UCB).

        Args:
            node: Current MCTS node.

        Returns:
            (str, MCTSNode): The move and corresponding child node.
        """
        best_score = -float('inf')
        best_move = None
        best_node = None

        total_visits = sum(self.N[move] for move in node.children)

        for move, child in node.children.items():
            ucb_score = self.ucb(node, move, total_visits)

            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_node = child

        return best_move, best_node

    def ucb(self, node, move, total_visits):
        """
        Compute the Upper Confidence Bound (UCB) score for a move.

        Args:
            node: Current MCTS node.
            move: Move being evaluated.
            total_visits: Total visits of all child nodes.

        Returns:
            float: UCB score.
        """
        q_value = self.Q[move]
        visit_count = self.N[move]
        prior_prob = self.P.get(move, 0)

        if visit_count == 0:
            return float('inf')  # Explore unvisited moves

        # UCB formula: Q + c_puct * P * sqrt(parent visits) / (1 + N)
        ucb_score = q_value + self.c_puct * prior_prob * math.sqrt(total_visits) / (1 + visit_count)
        return ucb_score
