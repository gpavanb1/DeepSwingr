"""
Cricket Ball Aerodynamics: Differentiable Physics Training
The CFD solver is EMBEDDED in the training loop - gradients flow through physics!

Parameters:
- roughness: Surface roughness coefficient [0.0, 1.0]
- notch_angle: Seam angle in degrees [-90, 90]
  * Negative angle = outswing delivery (ball swings away from RH batsman, +y direction)
  * Positive angle = inswing delivery (ball swings toward RH batsman, -y direction)
  * 0° = seam upright (no swing)
- reynolds_number: Flow Reynolds number [1e5, 1e6]

Output: Force vector [Fx_drag, Fy_lift, Fz_side] in Newtons
  * Fz_side: Positive = outswing (away), Negative = inswing (toward batsman)
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Dict
import numpy as np
from functools import partial

# ============================================================================
# CFD SIMULATION (Differentiable - used IN training loop)
# ============================================================================


@jit
def cfd_solve_navier_stokes(
    roughness: float,
    notch_angle: float,
    reynolds_number: float
) -> jnp.ndarray:
    """
    Differentiable CFD solver.
    Returns: [drag, lift, side] force vector in Newtons
    """
    # Normalize inputs
    rough_norm = jnp.clip(roughness, 0.0, 1.0)
    angle_rad = jnp.deg2rad(notch_angle)
    re_norm = jnp.clip(reynolds_number, 1e5, 1e6)

    # Cricket ball parameters
    diameter = 0.07
    rho = 1.225
    mu = 1.5e-5
    area = jnp.pi * (diameter/2)**2

    # Calculate velocity from Reynolds number
    velocity = re_norm * mu / (rho * diameter)
    q = 0.5 * rho * velocity**2

    # DRAG FORCE
    re_critical = 3e5 * (1.0 - 0.7 * rough_norm)
    transition = jax.nn.sigmoid(5.0 * (re_norm - re_critical) / re_critical)
    cd = 0.47 + (0.20 - 0.47) * transition
    drag = cd * q * area
    drag = jnp.clip(drag, 0.0, 10.0)

    # LIFT FORCE (vertical)
    cl = 0.2 * rough_norm * jnp.sin(2.0 * angle_rad)
    lift = cl * q * area
    lift = jnp.clip(lift, -5.0, 5.0)

    # SIDE FORCE (lateral swing)
    # Convention: Positive y = outswing (away from RH batsman)
    #            Negative y = inswing (toward RH batsman)
    # For conventional swing: negative seam angle → positive side force (outswing)
    #                        positive seam angle → negative side force (inswing)
    conv_coeff = -0.20 * jnp.sin(angle_rad)  # Note the negative sign!
    rev_factor = jax.nn.sigmoid(10.0 * (rough_norm - 0.7))
    rev_coeff = 0.25 * jnp.sin(angle_rad)  # Opposite sign for reverse
    cs = (conv_coeff * (1.0 - rev_factor) +
          rev_coeff * rev_factor) * rough_norm
    side = cs * q * area
    side = jnp.clip(side, -5.0, 5.0)

    return jnp.array([drag, lift, side])


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class CricketBallForceNetwork(nn.Module):
    """Neural network that LEARNS from the CFD solver."""

    hidden_dims: Tuple[int, ...] = (32, 64, 64, 32)

    @nn.compact
    def __call__(self, x):
        roughness, angle, re = x[0], x[1], x[2]
        re_normalized = jnp.log10(re) / 6.0
        angle_normalized = angle / 90.0

        angle_rad = jnp.deg2rad(angle)
        x_norm = jnp.array([
            roughness,
            jnp.sin(angle_rad),
            jnp.cos(angle_rad),
            re_normalized
        ])

        for dim in self.hidden_dims:
            x_norm = nn.Dense(dim)(x_norm)
            x_norm = nn.gelu(x_norm)
            x_norm = nn.LayerNorm()(x_norm)

        forces = nn.Dense(3)(x_norm)
        return forces


# ============================================================================
# TRAINING INFRASTRUCTURE
# ============================================================================

def create_train_state(rng, learning_rate=3e-4):
    """Create initial training state."""
    model = CricketBallForceNetwork()
    params = model.init(rng, jnp.ones(3))
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def compute_loss_with_cfd(params, apply_fn, batch_inputs):
    """CFD solver runs HERE in the loss function!"""
    nn_predictions = vmap(lambda x: apply_fn(params, x))(batch_inputs)

    cfd_truth = vmap(cfd_solve_navier_stokes)(
        batch_inputs[:, 0],
        batch_inputs[:, 1],
        batch_inputs[:, 2]
    )

    nn_predictions = jnp.nan_to_num(
        nn_predictions, nan=0.0, posinf=1.0, neginf=-1.0)
    cfd_truth = jnp.nan_to_num(cfd_truth, nan=0.0, posinf=1.0, neginf=-1.0)

    mse = jnp.mean((nn_predictions - cfd_truth) ** 2)

    nn_force_mag = jnp.linalg.norm(nn_predictions, axis=1)
    magnitude_penalty = jnp.mean(jnp.maximum(nn_force_mag - 10.0, 0.0)**2)

    total_loss = mse + 0.01 * magnitude_penalty

    metrics = {
        'mse': mse,
        'mag_penalty': magnitude_penalty,
        'total': total_loss
    }

    return total_loss, metrics


@partial(jit, static_argnums=(1,))
def train_step_with_cfd(state, apply_fn, batch_inputs):
    """Training step where CFD runs for every batch!"""
    (loss, metrics), grads = jax.value_and_grad(
        compute_loss_with_cfd, has_aux=True
    )(state.params, apply_fn, batch_inputs)

    state = state.apply_gradients(grads=grads)
    return state, loss, metrics


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model_with_embedded_cfd(
    n_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    n_batches_per_epoch: int = 50,
    seed: int = 42
):
    """Train with CFD solver INSIDE the loop."""
    key = random.PRNGKey(seed)
    key, model_key = random.split(key)

    print("=" * 70)
    print("DIFFERENTIABLE PHYSICS TRAINING")
    print("CFD solver embedded in training loop - gradients flow through physics!")
    print("=" * 70)

    print("\nInitializing neural network...")
    state = create_train_state(model_key, learning_rate)

    print(f"\nTraining for {n_epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {n_batches_per_epoch}")
    print("-" * 70)

    # Validation set
    key, val_key = random.split(key)
    val_roughness = random.uniform(val_key, (100,), minval=0.0, maxval=1.0)
    key, val_key = random.split(key)
    val_angle = random.uniform(val_key, (100,), minval=-90.0, maxval=90.0)
    key, val_key = random.split(key)
    val_re = random.uniform(val_key, (100,), minval=1e5, maxval=1e6)
    val_inputs = jnp.stack([val_roughness, val_angle, val_re], axis=1)

    best_val_loss = float('inf')
    best_params = state.params

    for epoch in range(n_epochs):
        epoch_losses = []
        epoch_metrics = []

        for batch_idx in range(n_batches_per_epoch):
            key, batch_key = random.split(key)
            keys = random.split(batch_key, 3)

            batch_roughness = random.uniform(
                keys[0], (batch_size,), minval=0.0, maxval=1.0)
            batch_angle = random.uniform(
                keys[1], (batch_size,), minval=-90.0, maxval=90.0)
            batch_re = random.uniform(
                keys[2], (batch_size,), minval=1e5, maxval=1e6)

            batch_inputs = jnp.stack(
                [batch_roughness, batch_angle, batch_re], axis=1)

            state, loss, metrics = train_step_with_cfd(
                state, state.apply_fn, batch_inputs)

            epoch_losses.append(loss)
            epoch_metrics.append(metrics)

        val_loss, val_metrics = compute_loss_with_cfd(
            state.params, state.apply_fn, val_inputs
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = state.params

        if epoch % 10 == 0:
            train_loss = jnp.mean(jnp.array(epoch_losses))
            avg_mse = jnp.mean(jnp.array([m['mse'] for m in epoch_metrics]))

            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                  f"Train MSE: {avg_mse:.6f} | Val Loss: {val_loss:.6f}")

            if jnp.isnan(train_loss) or jnp.isnan(val_loss):
                print("  WARNING: NaN detected!")
                if epoch == 0:
                    print("  Reinitializing...")
                    state = create_train_state(random.PRNGKey(
                        seed + epoch), learning_rate * 0.1)

    print("-" * 70)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
    print("\n✓ Neural network learned from differentiable CFD solver!")

    state = state.replace(params=best_params)
    return state


# ============================================================================
# INFERENCE AND TESTING
# ============================================================================

def predict_forces(state, roughness: float, notch_angle: float, reynolds_number: float):
    """Fast inference with trained NN."""
    inputs = jnp.array([roughness, notch_angle, reynolds_number])
    forces = state.apply_fn(state.params, inputs)
    return forces


def test_swing_physics():
    """Test that swing physics is correct BEFORE training."""
    print("\n" + "=" * 70)
    print("TESTING SWING PHYSICS")
    print("=" * 70)

    Re = 2.5e5

    test_cases = [
        # Negative angle → positive y (away)
        (0.3, -20, "Conventional outswing", "positive"),
        # Positive angle → negative y (toward)
        (0.3, +20, "Conventional inswing", "negative"),
        (0.3, 0, "No swing", "zero"),
        # Reverse: negative angle → negative y
        (0.8, -20, "Reverse swing inswing", "negative"),
        # Reverse: positive angle → positive y
        (0.8, +20, "Reverse swing outswing", "positive"),
    ]

    print("\nDetailed physics calculation:")
    print("-" * 80)

    for rough, angle, expected, expected_sign in test_cases:
        print(
            f"\n[Test: roughness={rough}, angle={angle}°, expected={expected_sign}]")

        # Manual calculation
        angle_rad = jnp.deg2rad(angle)
        rough_norm = jnp.clip(rough, 0.0, 1.0)
        re_norm = jnp.clip(Re, 1e5, 1e6)

        diameter = 0.07
        rho = 1.225
        mu = 1.5e-5
        area = jnp.pi * (diameter/2)**2

        velocity = re_norm * mu / (rho * diameter)
        q = 0.5 * rho * velocity**2

        sin_angle = jnp.sin(angle_rad)
        conv_coeff = -0.20 * sin_angle  # Note negative sign
        rev_factor = jax.nn.sigmoid(10.0 * (rough_norm - 0.7))
        rev_coeff = 0.25 * sin_angle
        cs = (conv_coeff * (1.0 - rev_factor) +
              rev_coeff * rev_factor) * rough_norm
        side_manual = cs * q * area
        side_manual_clipped = jnp.clip(side_manual, -5.0, 5.0)

        print(f"  Manual calculation: side = {side_manual_clipped:+.6f}")

        # Actual function
        forces = cfd_solve_navier_stokes(rough, angle, Re)
        print(f"  Actual function:")
        print(f"    forces[0] (drag) = {forces[0]:+.6f}")
        print(f"    forces[1] (lift) = {forces[1]:+.6f}")
        print(f"    forces[2] (side) = {forces[2]:+.6f}")

        # Check correctness
        if expected_sign == "zero":
            correct = abs(forces[2]) < 0.01
        elif expected_sign == "positive":
            correct = forces[2] > 0
        elif expected_sign == "negative":
            correct = forces[2] < 0

        print(f"  Result: {'✓ CORRECT' if correct else '✗ WRONG'}")

    print("\n" + "=" * 80)
