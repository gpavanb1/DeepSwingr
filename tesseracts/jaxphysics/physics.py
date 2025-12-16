"""
physics.py — Differentiable 2D cricket ball swing with jax-cfd 0.2.1
"""
import jax_cfd.base as cfd
import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
import flax.linen as nn
from flax.training import train_state
import optax

# ============================================================================
# PARAMETERS
# ============================================================================
GRID_SIZE = 128
DOMAIN_SIZE = 2.0
DT = 0.01
BASE_VISCOSITY = 1e-3
INJECTION_MAG = 0.35

# Coordinates + ball mask
x = jnp.linspace(-DOMAIN_SIZE, DOMAIN_SIZE, GRID_SIZE, endpoint=False)
y = jnp.linspace(-DOMAIN_SIZE, DOMAIN_SIZE, GRID_SIZE, endpoint=False)
X, Y = jnp.meshgrid(x, y, indexing='ij')
RADIUS = 0.07
BALL_MASK = (X**2 + Y**2) <= RADIUS**2

# ============================================================================
# CFD SETUP – jax-cfd 0.2.1
# ============================================================================

grid = cfd.grids.Grid(
    shape=(GRID_SIZE, GRID_SIZE),
    domain=((-DOMAIN_SIZE, DOMAIN_SIZE), (-DOMAIN_SIZE, DOMAIN_SIZE))
)

# Periodic BCs using HomogeneousBoundaryConditions (standard for periodic in demos)
bc = cfd.boundaries.HomogeneousBoundaryConditions(
    (
        (cfd.boundaries.BCType.PERIODIC, cfd.boundaries.BCType.PERIODIC),
        (cfd.boundaries.BCType.PERIODIC, cfd.boundaries.BCType.PERIODIC),
    )
)


def injection_field(angle_deg):
    angle_rad = jnp.deg2rad(angle_deg)
    inj_x = jnp.zeros_like(X)
    inj_y = INJECTION_MAG * jnp.sin(angle_rad) * BALL_MASK.astype(jnp.float32)
    inj = jnp.stack([inj_x, inj_y], axis=-1)

    offsets = ((1.0, 0.5), (0.5, 1.0))
    return tuple(
        cfd.grids.GridVariable(
            array=cfd.grids.GridArray(
                inj[..., i], offset=offsets[i], grid=grid),
            bc=bc
        ) for i in range(2)
    )


@jit
def cfd_solve_navier_stokes(roughness, notch_angle, reynolds_number, n_steps=25):
    """Run CFD with a freestream velocity computed from Reynolds number."""
    # Physical parameters
    diameter = 2 * RADIUS
    nu = BASE_VISCOSITY

    # Compute freestream velocity
    U_infty = (reynolds_number * nu) / diameter

    # Build solver
    step_fn = cfd.equations.semi_implicit_navier_stokes(
        density=1.0,
        viscosity=nu,
        dt=DT,
        grid=grid
    )

    # Initialize uniform background flow (freestream in +x direction)
    vx0 = jnp.full((GRID_SIZE, GRID_SIZE), U_infty)
    vy0 = jnp.zeros_like(vx0)

    velocity = tuple(
        cfd.grids.GridVariable(
            array=cfd.grids.GridArray(
                arr, offset=offset, grid=grid
            ),
            bc=bc
        ) for arr, offset in zip((vx0, vy0), ((1.0, 0.5), (0.5, 1.0)))
    )

    def body(v, _):
        inj = injection_field(notch_angle)
        forced = tuple(
            cfd.grids.GridVariable(
                array=cfd.grids.GridArray(
                    v_i.array.data + DT * inj_i.array.data,
                    offset=v_i.array.offset,
                    grid=grid
                ),
                bc=bc
            ) for v_i, inj_i in zip(v, inj)
        )
        return step_fn(forced), None

    final_v, _ = jax.lax.scan(body, velocity, None, length=n_steps)

    u = final_v[0].array.data
    v = final_v[1].array.data

    Drag = -jnp.sum((u - U_infty) * BALL_MASK)
    Lift = jnp.sum(v * BALL_MASK)
    Side = 0.0

    return jnp.array([Drag, Lift, Side])


# ============================================================================
# NEURAL NETWORK
# ============================================================================


class CricketBallForceNetwork(nn.Module):
    hidden_dims: tuple = (64, 128, 128, 64)

    @nn.compact
    def __call__(self, x):
        rough, ang, re = x[0], x[1], x[2]
        ang_rad = jnp.deg2rad(ang)
        inp = jnp.array([rough,
                         jnp.sin(ang_rad),
                         jnp.cos(ang_rad),
                         jnp.log10(re) / 6.0])
        h = inp
        for d in self.hidden_dims:
            h = nn.Dense(d)(h)
            h = nn.gelu(h)
            h = nn.LayerNorm()(h)
        return nn.Dense(3)(h)

# ============================================================================
# TRAINING
# ============================================================================


def create_train_state(rng, lr=3e-4):
    model = CricketBallForceNetwork()
    params = model.init(rng, jnp.ones(3))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_loss_with_cfd(params, apply_fn, batch):
    pred = vmap(apply_fn, in_axes=(None, 0))(params, batch)
    truth = vmap(cfd_solve_navier_stokes)(
        batch[:, 0], batch[:, 1], batch[:, 2])
    mse = jnp.mean((pred - truth) ** 2)
    return mse, {'mse': mse}


@partial(jit, static_argnums=(1,))
def train_step_with_cfd(state, apply_fn, batch):
    (loss, metrics), grads = jax.value_and_grad(compute_loss_with_cfd, has_aux=True)(
        state.params, apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics
