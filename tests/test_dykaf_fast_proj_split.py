import ast
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.modules.setdefault("wandb", types.SimpleNamespace(run=None))

from optim.dykaf_with_parallel_proj_split import DyKAF, proj_split


def paper_kron_proj_split(L, R, g, beta):
    L = L * beta**0.5
    R = R * beta**0.5
    g = g * (1 - beta) ** 0.5

    r = torch.linalg.norm(R)
    R = R / r
    C = g @ R @ g.T
    L_hat = L * r + C
    S_hat = torch.linalg.norm(L_hat)
    L_next = L_hat / S_hat
    S_tilde = S_hat - torch.sum(L_next * C)
    R_hat = R * S_tilde + g.T @ L_next @ g
    S_next = torch.linalg.norm(R_hat)
    return L_next * S_next**0.5, R_hat / S_next**0.5


class FastProjectorSplitTest(unittest.TestCase):
    def test_matches_paper_algorithm_for_initialized_factors(self):
        torch.manual_seed(7)
        L = torch.randn(5, 5, dtype=torch.float64)
        R = torch.randn(3, 3, dtype=torch.float64)
        g = torch.randn(5, 3, dtype=torch.float64)
        beta = 0.95

        expected_L, expected_R = paper_kron_proj_split(L, R, g, beta)
        actual_L, actual_R = proj_split(
            L.clone(),
            R.clone(),
            g.clone(),
            beta=beta,
            init="kron",
            factors_initialized=True,
        )

        torch.testing.assert_close(actual_L, expected_L)
        torch.testing.assert_close(actual_R, expected_R)

    def test_default_dykaf_cli_uses_fast_paper_implementation(self):
        tree = ast.parse((REPO_ROOT / "src" / "main.py").read_text())
        imports = {
            alias.asname or alias.name: node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        self.assertEqual(
            imports["DyKAF"],
            "optim.dykaf_with_parallel_proj_split",
        )

    def test_optimizer_keeps_finite_fast_factors_across_steps(self):
        torch.manual_seed(11)
        parameter = torch.nn.Parameter(torch.randn(5, 3))
        optimizer = DyKAF(
            [parameter],
            lr=1e-3,
            betas=(0.9, 0.95),
            shampoo_beta=0.95,
            precondition_frequency=2,
            init="kron",
            adam_rank_one=False,
        )

        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            parameter.square().mean().backward()
            optimizer.step()

        state = optimizer.state[parameter]
        self.assertTrue(state["dykaf_factors_initialized"])
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertTrue(all(torch.isfinite(factor).all() for factor in state["GG"]))


if __name__ == "__main__":
    unittest.main()
