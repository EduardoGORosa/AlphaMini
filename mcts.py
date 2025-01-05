# mcts_improved.py
import math
import numpy as np
import torch
from collections import defaultdict

class MCTSNode:
    """
    Represents a node in the MCTS tree.
    """
    def __init__(self, env, parent=None, prior=0.0):
        self.env = env       # ChessEnv instance (or similar) with the current position
        self.parent = parent
        self.children = {}   # dict: move_uci -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.expanded = False

    def is_leaf(self):
        return len(self.children) == 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """
    MCTS with:
      1) Re-rooting each move (so we don't discard the tree),
      2) Dirichlet noise at the root for exploration,
      3) Partial batching of expansions.
    """
    def __init__(self, model, move_to_index, c_puct=1.4, dirichlet_alpha=0.3, dirichlet_eps=0.25):
        """
        Args:
          model: your AlphaZeroNet in eval mode
          move_to_index: dict mapping move_uci -> int
          c_puct: exploration constant
          dirichlet_alpha: for root dirichlet noise
          dirichlet_eps: fraction of noise to mix in
        """
        self.model = model
        self.move_to_index = move_to_index
        self.c_puct = c_puct

        # Dirichlet noise params
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

        # We'll keep track of the current root node
        self.root = None

    def initialize_root(self, env):
        """
        Called at the start of a game to create an initial root node.
        """
        self.root = MCTSNode(env.clone(), parent=None, prior=0.0)
        # Expand
        self._expand_node(self.root, add_dirichlet=True)

    def re_root(self, move_uci):
        """
        After we pick a move from the root, we re-root the tree to that child.
        If the child doesn't exist, create a new node from scratch.
        """
        if move_uci in self.root.children:
            child = self.root.children[move_uci]
            child.parent = None
            self.root = child
        else:
            # child wasn't expanded for some reason, create new
            env_copy = self.root.env.clone()
            env_copy.step(move_uci)
            self.root = MCTSNode(env_copy, parent=None, prior=0.0)
            self._expand_node(self.root, add_dirichlet=False)

    def run_simulations(self, num_simulations=100, batch_size=8):
        """
        Perform MCTS simulations from the current root.
        We'll gather leaves in small batches and evaluate them in one net forward pass.
        """
        for _ in range(num_simulations // batch_size):
            leaves = []
            for _ in range(batch_size):
                leaf = self._select_leaf(self.root)
                leaves.append(leaf)

            # Evaluate these leaves in ONE forward pass
            self._expand_and_evaluate_batch(leaves)

        # If remainder
        remainder = num_simulations % batch_size
        if remainder > 0:
            leaves = []
            for _ in range(remainder):
                leaf = self._select_leaf(self.root)
                leaves.append(leaf)
            self._expand_and_evaluate_batch(leaves)

    def get_move_visit_counts(self):
        """
        After run_simulations, return {move_uci: visit_count} for the root's children.
        """
        visit_counts = {}
        for mv, child in self.root.children.items():
            visit_counts[mv] = child.visit_count
        return visit_counts

    ################################################################
    # Internal MCTS routines
    ################################################################
    def _select_leaf(self, node):
        """
        Descend the tree using UCB until we reach a leaf or a node that wasn't expanded.
        """
        current = node
        while not current.env.is_game_over() and current.expanded and not current.is_leaf():
            best_score = -float('inf')
            best_move = None
            best_child = None

            total_visits = sum(c.visit_count for c in current.children.values())
            for mv, child in current.children.items():
                q = child.value()
                u = self.c_puct * child.prior * math.sqrt(current.visit_count) / (1 + child.visit_count)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_move = mv
                    best_child = child

            current = best_child

        return current

    def _expand_and_evaluate_batch(self, leaves):
        """
        For each leaf:
          - If game over => backprop final result
          - Else => expand + evaluate net in a single forward pass
        """
        # Separate out those that need net evaluation
        leaves_to_expand = []
        states = []
        for leaf in leaves:
            if leaf.env.is_game_over():
                outcome = leaf.env.get_result()
                leaf.visit_count += 1
                leaf.value_sum += outcome
                self._backpropagate(leaf, outcome)
            else:
                leaves_to_expand.append(leaf)
                states.append(leaf.env.get_state_representation())

        if not leaves_to_expand:
            return

        # Batch forward pass
        device = next(self.model.parameters()).device
        batch_tensor = [torch.tensor(st, dtype=torch.float32) for st in states]
        batch_tensor = torch.stack(batch_tensor, dim=0)  # [batch,13,8,8]
        batch_tensor = batch_tensor.to(device)

        with torch.no_grad():
            policy_logits, value_out = self.model(batch_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = value_out.squeeze(dim=1).cpu().numpy()

        # Expand each leaf
        for i, leaf in enumerate(leaves_to_expand):
            val = float(values[i])
            leaf.visit_count += 1
            leaf.value_sum += val

            self._expand_node(leaf, add_dirichlet=False, policy_array=policy_probs[i])
            self._backpropagate(leaf, val)

    def _expand_node(self, node, add_dirichlet=False, policy_array=None):
        """
        Expand a node by enumerating legal moves, set child prior from policy.
        If add_dirichlet=True, we apply Dirichlet noise at the root for exploration.
        """
        if node.env.is_game_over():
            node.expanded = True
            return

        if policy_array is None:
            # Single forward pass if not given
            state_tensor = torch.tensor(node.env.get_state_representation(), dtype=torch.float32).unsqueeze(0)
            device = next(self.model.parameters()).device
            state_tensor = state_tensor.to(device)
            with torch.no_grad():
                p_logits, val = self.model(state_tensor)
            policy_probs = torch.softmax(p_logits, dim=1).cpu().numpy()[0]
        else:
            policy_probs = policy_array

        legal_moves = node.env.get_legal_moves()
        priors = []
        for mv in legal_moves:
            idx = self.move_to_index.get(mv, None)
            p = policy_probs[idx] if idx is not None else 1e-6
            priors.append((mv, p))

        # If root node, add Dirichlet noise
        if node == self.root and add_dirichlet and len(priors) > 0:
            alpha = self.dirichlet_alpha
            eps = self.dirichlet_eps
            noise = np.random.dirichlet([alpha] * len(priors))
            new_priors = []
            for i, (mv, old_p) in enumerate(priors):
                new_p = (1 - eps) * old_p + eps * noise[i]
                new_priors.append((mv, new_p))
            priors = new_priors

        # Create children
        for mv, prior in priors:
            child_env = node.env.clone()
            child_env.step(mv)
            child_node = MCTSNode(child_env, parent=node, prior=prior)
            node.children[mv] = child_node

        node.expanded = True

    def _backpropagate(self, node, value):
        """
        Backprop 'value' from leaf up to root.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent
