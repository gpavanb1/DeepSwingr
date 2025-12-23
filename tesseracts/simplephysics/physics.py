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
from jax import jit
import flax.linen as nn
from typing import Tuple

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
