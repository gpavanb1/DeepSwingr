"""
Training script for the simplephysics neural network model.

This script trains the CricketBallForceNetwork defined in tesseracts.simplephysics.physics
and saves the trained weights checkpoint into the tesseracts/simplephysics directory
for use by the deployed simplephysics tesseract for inference.
"""

import os
import jax
from jax import random
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import jax.numpy as jnp

from physics import CricketBallForceNetwork, compute_loss_with_cfd, train_step_with_cfd


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
        tx=tx,
    )


def train_model(
    n_epochs=100,
    batch_size=16,
    learning_rate=3e-4,
    n_batches_per_epoch=50,
    seed=42,
    weights_dir="./weights",
    weights_filename="simplephysics_weights.msgpack"
):
    """Train the model and save the final weights for inference."""

    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, weights_filename)

    key = random.PRNGKey(seed)
    key, model_key = random.split(key)

    print("=" * 70)
    print("Training CricketBallForceNetwork")
    print("=" * 70)

    state = create_train_state(model_key, learning_rate)

    best_val_loss = float("inf")
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

        # Validation set per epoch
        key, val_key = random.split(key)
        val_roughness = random.uniform(val_key, (100,), minval=0.0, maxval=1.0)
        key, val_key = random.split(key)
        val_angle = random.uniform(val_key, (100,), minval=-90.0, maxval=90.0)
        key, val_key = random.split(key)
        val_re = random.uniform(val_key, (100,), minval=1e5, maxval=1e6)
        val_inputs = jnp.stack([val_roughness, val_angle, val_re], axis=1)

        val_loss, val_metrics = compute_loss_with_cfd(
            state.params, state.apply_fn, val_inputs
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = state.params

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            train_loss = jnp.mean(jnp.array(epoch_losses))
            avg_mse = jnp.mean(jnp.array([m['mse'] for m in epoch_metrics]))
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                  f"Train MSE: {avg_mse:.6f} | Val Loss: {val_loss:.6f}")

    print("-" * 70)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")

    # Save best parameters to a single weights file for inference
    import flax.serialization
    params_bytes = flax.serialization.to_bytes(best_params)
    with open(weights_path, "wb") as f:
        f.write(params_bytes)

    return best_params


if __name__ == "__main__":
    train_model()
