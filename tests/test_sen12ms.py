# SPDX-License-Identifier: MIT
"""Unit test of the SEN12MS certificate (no data needed): on random merged counts, the closed-form certificate of
pipeline/07d_sen12ms.py agrees with program (1) solved exactly, every stored rival state lies in the fiber, and on a
certified value whose 17 counts vary, the second state lies in the fiber and still elects the label."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sen12ms_step", REPO / "pipeline" / "07d_sen12ms.py")
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)

GROUPS = [[0, 1, 2, 3, 4], [5, 6], [7, 8], [9], [10], [11, 13], [12], [14], [15], [16]]


def main() -> int:
    W = np.zeros((len(GROUPS), 17), dtype=np.int64)
    gid = np.zeros(17, dtype=int)
    for g, mem in enumerate(GROUPS):
        W[g, mem] = 1
        gid[mem] = g
    support = list(range(17))
    rng = random.Random(7)
    seen = {True: 0, False: 0}
    second = 0
    for _ in range(300):
        b = np.zeros(17, dtype=np.int64)
        for k in rng.sample(range(17), rng.randint(1, 5)):
            b[k] = rng.randint(1, 30)
        f = int(b.argmax())
        if (b == b[f]).sum() > 1:
            continue
        v = W @ b
        closed = step.certifiable(v, GROUPS, gid, set(support), f)
        exact = step.milp_certifiable(v, W, f, support)
        assert closed == exact, (b.tolist(), closed, exact)
        if not closed:
            x, r = step.rival_state(b, v, GROUPS, gid, f)
            assert np.array_equal(W @ x, v) and x.min() >= 0 and x[r] >= x[f]
        elif any(v[g] > 0 and len(GROUPS[g]) > 1 for g in range(len(GROUPS))):
            x = step.second_state(b, v, GROUPS, set(support))
            assert np.array_equal(W @ x, v) and x.min() >= 0 and not np.array_equal(x, b)
            assert all(x[f] > x[r] for r in range(17) if r != f)
            second += 1
        seen[closed] += 1
    assert seen[True] > 0 and seen[False] > 0 and second > 0
    print(f"test_sen12ms: ok ({seen[True]} certifiable, {second} of them with varying counts, {seen[False]} open values)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
