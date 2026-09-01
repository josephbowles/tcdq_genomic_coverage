#!/usr/bin/env python
"""Train an IQP Born machine on the PennyLane genomic SNV dataset and score it by sampling.

Self-contained: downloads the data, trains the MMD loss with pennylane.labs.tcdq,
samples the trained circuit, and writes a loss/coverage figure.

The data is the 1000 Genomes SNV set served by PennyLane datasets: 5,008 haplotypes over
805 loci, of which the first ``--n-qubits`` are modelled. 

Usage::

    pip install "pennylane>=0.46" jax jaxopt optax matplotlib numpy
    python standalone_genomic_iqp.py
"""

from __future__ import annotations

import argparse
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pennylane as qp  # noqa: E402
from pennylane.labs.tcdq import (  # noqa: E402
    CircuitConfig,
    MMDConfig,
    TrainingOptions,
    create_local_gates,
    mmd_loss,
    training_iterator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=20)
    parser.add_argument("--eps", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shots", type=int, default=5008)

    parser.add_argument("--max-weight", type=int, default=4)
    parser.add_argument("--init-scale", type=float, default=0.001)
    parser.add_argument("--mean-pauli-weight", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-ops", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--unroll", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=100)

    parser.add_argument("--smooth-window", type=int, default=500)
    parser.add_argument("--out", default="standalone_genomic_iqp.png")
    return parser.parse_args()


def pack(bits: np.ndarray, n_qubits: int) -> np.ndarray:
    """Pack bitstrings into int64 keys with wire 0 as the most significant bit."""
    return np.asarray(bits, dtype=np.int64) @ (1 << np.arange(n_qubits - 1, -1, -1)).astype(
        np.int64
    )


def unpack(keys: np.ndarray, n_qubits: int) -> np.ndarray:
    shifts = np.arange(n_qubits - 1, -1, -1)
    return ((np.asarray(keys, dtype=np.int64)[:, None] >> shifts) & 1).astype(np.int8)


def sigma_for_mean_weight(mean_weight: float, n_qubits: int) -> float:
    """Bandwidth whose sampled observables have the given mean Pauli weight.

    The kernel includes each qubit with probability ``(1 - exp(-1/(2 sigma^2)))/2``.
    """
    return float(1.0 / np.sqrt(-2.0 * np.log1p(-2.0 * mean_weight / n_qubits)))


def load_task(n_qubits: int, eps: float, seed: int) -> dict:
    """Genomic task: the valid set is the deduplicated observed set on the first N loci."""
    dataset = qp.data.load("other", name="genomic")[0]
    data = np.vstack([np.asarray(dataset.train), np.asarray(dataset.test)]).astype(np.int8)
    data = data[:, :n_qubits]

    valid_keys, counts = np.unique(pack(data, n_qubits), return_counts=True)
    target_probs = counts / counts.sum()

    rng = np.random.default_rng(seed)
    n_train = int(eps * len(valid_keys))
    train = data[rng.choice(len(data), n_train, replace=False)]
    train_keys = np.unique(pack(train, n_qubits))

    return {
        "n_qubits": n_qubits,
        "train": train,
        "valid_keys": valid_keys,
        "target_probs": target_probs,
        "unseen_keys": np.setdiff1d(valid_keys, train_keys),
    }


def init_params(gates: dict, task: dict, scale: float, seed: int) -> np.ndarray:
    """Gaussian parameters with the weight-1 gates centred on the training marginals.

    At theta = 0 the circuit is the identity, so the model would start as a point mass
    on |0...0>. With the higher-weight gates near zero the circuit factorizes and a lone
    qubit gives p(1) = sin^2(theta), so arcsin(sqrt(marginal)) reproduces the marginals.
    """
    params = np.random.default_rng(seed).normal(0.0, scale, size=len(gates))
    angles = np.arcsin(np.sqrt(np.clip(task["train"].mean(axis=0), 0.0, 1.0)))
    for idx, wire_lists in gates.items():
        wires = wire_lists[0]
        if len(wires) == 1:
            params[idx] += angles[wires[0]]
    return params


def fwht(vector: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform, ``out[A] = sum_x v[x] (-1)^{A.x}``."""
    out = np.array(vector)
    n = out.size
    block = 1
    while block < n:
        out = out.reshape(-1, 2, block)
        lo, hi = out[:, 0, :].copy(), out[:, 1, :].copy()
        out[:, 0, :], out[:, 1, :] = lo + hi, lo - hi
        out = out.reshape(n)
        block *= 2
    return out


def iqp_probs(params: np.ndarray, gates: dict, n_qubits: int) -> np.ndarray:
    """Exact output distribution of ``H^n D(theta) H^n |0^n>``, indexed by bitstring key.

    D is diagonal with entries exp(i f(x)) and f is the Walsh transform of the gate
    coefficients, so the circuit is two Walsh transforms regardless of the gate count.
    """
    n_states = 1 << n_qubits
    coeffs = np.zeros(n_states, dtype=np.float64)
    thetas = np.asarray(params, dtype=np.float64)

    for idx, wire_lists in gates.items():
        for wires in wire_lists:
            mask = 0
            for wire in wires:
                mask |= 1 << (n_qubits - 1 - wire)
            coeffs[mask] += thetas[idx]

    return np.abs(fwht(np.exp(1j * fwht(coeffs))) / n_states) ** 2


def sample(params: np.ndarray, gates: dict, n_qubits: int, shots: int, seed: int) -> np.ndarray:
    probs = iqp_probs(params, gates, n_qubits)
    return np.random.default_rng(seed).choice(probs.size, size=shots, p=probs)


def coverage(sample_keys: np.ndarray, task: dict) -> float:
    """Fraction of the unseen valid set reached by the sampled strings."""
    hit = np.intersect1d(np.unique(sample_keys), task["unseen_keys"], assume_unique=True)
    return len(hit) / len(task["unseen_keys"])


def train(task: dict, gates: dict, params: np.ndarray, args: argparse.Namespace, sigma: float):
    """Adam on the sampled-spectrum MMD^2 estimate, resampling the target each step.

    The estimator's data term is a U-statistic that is unbiased only when the target
    data is redrawn at every evaluation; with a fixed training set the loss converges
    below zero.
    """
    circuit_config = CircuitConfig(
        gates=gates,
        n_qubits=args.n_qubits,
        n_samples=args.n_samples,
        key=jax.random.PRNGKey(args.seed),
    )
    mmd_config = MMDConfig(bandwidth=sigma, n_ops=args.n_ops)
    n_draw = len(task["train"])

    def loss_fn(params, target_data, key):
        boot_key, loss_key = jax.random.split(key)
        pool = jnp.asarray(target_data)
        rows = jax.random.randint(boot_key, (n_draw,), 0, pool.shape[0])
        return mmd_loss(params, circuit_config, mmd_config, jnp.take(pool, rows, axis=0), loss_key)

    iterator = training_iterator(
        optimizer="Adam",
        loss=loss_fn,
        stepsize=args.lr,
        loss_kwargs={
            "params": np.asarray(params),
            "target_data": np.asarray(task["train"]),
            "key": jax.random.PRNGKey(args.seed + 1),
        },
        options=TrainingOptions(
            unroll_steps=args.unroll,
            # A window longer than the run disables the built-in early stop.
            convergence_interval=args.steps + 1,
            random_state=args.seed,
        ),
    )

    n_batches = math.ceil(args.steps / args.unroll)
    losses, checkpoints = [], []

    for i, batch in enumerate(iterator):
        if i >= n_batches:
            break
        losses.append(np.asarray(batch.losses, dtype=float))
        params = batch.params
        step = min((i + 1) * args.unroll, args.steps)

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            keys = sample(np.asarray(params), gates, args.n_qubits, args.shots, args.seed)
            checkpoints.append((step, coverage(keys, task)))
            print(
                f"  step {step:>5}/{args.steps}  loss = {losses[-1][-1]:.4e}  "
                f"C = {checkpoints[-1][1]:.4f}",
                flush=True,
            )

    return np.asarray(params), np.concatenate(losses)[: args.steps], checkpoints


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median, truncated at the edges rather than padded.

    Padding would pin the ends of the curve to a single noisy sample.
    """
    half = window // 2
    return np.array(
        [np.median(values[max(0, i - half) : i + half + 1]) for i in range(len(values))]
    )


def plot(losses, checkpoints, args, sigma, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    series = rolling_median(losses, args.smooth_window)

    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, color="0.72", lw=0.8, zorder=1, label="MMD$^2$ per step")
    ax.plot(
        steps,
        series,
        color="0.12",
        lw=2.0,
        zorder=3,
        label=f"MMD$^2$ (rolling median, {args.smooth_window} steps)",
    )

    ax.set_yscale("log")
    positive = series[series > 0]
    ax.set_ylim(positive.min() * 0.5, np.max(losses) * 2.0)
    ax.set_xlabel("training step")
    ax.set_ylabel(f"MMD$^2$ training loss ($\\sigma={sigma:.3g}$)")
    ax.set_title(
        f"IQP Born machine, genomic task, $N={args.n_qubits}$, "
        f"$\\sigma={sigma:.3f}$ (mean Pauli weight {args.mean_pauli_weight:g})",
        fontsize=11,
    )

    handles, labels = ax.get_legend_handles_labels()
    if checkpoints:
        steps_c, cov = zip(*checkpoints)
        c_rand = 1.0 - (1.0 - 2.0**-args.n_qubits) ** args.shots
        twin = ax.twinx()
        twin.plot(steps_c, cov, color="crimson", marker="o", ms=4.5, lw=1.6, label="coverage $C$")
        twin.axhline(
            c_rand, color="crimson", ls="--", lw=1.2, label="random-bit floor $C_{\\rm rand}$"
        )
        twin.set_ylabel("coverage $C$", color="crimson")
        twin.tick_params(axis="y", labelcolor="crimson")
        twin.set_ylim(0, max(max(cov), c_rand) * 1.9)
        extra = twin.get_legend_handles_labels()
        handles, labels = handles + extra[0], labels + extra[1]

    ax.legend(handles, labels, loc="lower left", fontsize=8.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    sigma = sigma_for_mean_weight(args.mean_pauli_weight, args.n_qubits)

    task = load_task(args.n_qubits, args.eps, args.seed)
    gates = create_local_gates(args.n_qubits, max_weight=args.max_weight)
    params = init_params(gates, task, args.init_scale, args.seed)

    print(
        f"N={args.n_qubits}  |S|={len(task['valid_keys']):,}  |T|={len(task['train']):,}  "
        f"|U|={len(task['unseen_keys']):,}  params={len(gates):,}  sigma={sigma:.4f}"
    )

    params, losses, checkpoints = train(task, gates, params, args, sigma)

    keys = sample(params, gates, args.n_qubits, args.shots, args.seed + 1000)
    probs = iqp_probs(params, gates, args.n_qubits)
    valid = np.isin(keys, task["valid_keys"])
    new = ~np.isin(keys, pack(task["train"], args.n_qubits))

    q_valid = probs[task["valid_keys"]]
    kl = float(np.sum(task["target_probs"] * np.log(task["target_probs"] / q_valid)))
    c_rand = 1.0 - (1.0 - 2.0**-args.n_qubits) ** args.shots

    print(
        f"\ncoverage C   = {coverage(keys, task):.4f}   (random-bit floor {c_rand:.4f})\n"
        f"fidelity F   = {(valid & new).sum() / max(new.sum(), 1):.4f}\n"
        f"valid mass   = {q_valid.sum():.4f}\n"
        f"forward KL   = {kl:.4f}"
    )

    np.savez(
        args.out.replace(".png", ".npz"),
        params=params,
        losses=losses,
        checkpoints=np.asarray(checkpoints, dtype=float),
    )
    plot(losses, checkpoints, args, sigma, args.out)


if __name__ == "__main__":
    main()
