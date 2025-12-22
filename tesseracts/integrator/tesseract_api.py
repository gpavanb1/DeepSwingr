import jax
import jax.numpy as jnp
from typing import Any
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Float32, Differentiable, ShapeDType
from integrate import simulate_trajectory


class InputSchema(BaseModel):
    """Input schema for integrator tesseract"""
    initial_velocity: Differentiable[Float32] = Field(
        ..., description="Initial velocity in m/s [20.0, 50.0]")
    release_angle: Differentiable[Float32] = Field(
        ..., description="Release angle in degrees [0.0, 10.0]")
    roughness: Differentiable[Float32] = Field(
        ..., description="Ball surface roughness [0.0, 1.0]")
    seam_angle: Differentiable[Float32] = Field(
        ..., description="Seam angle in degrees [-90.0, 90.0]")
    physics_url: str = Field(
        default="http://simplephysics:8000",
        description="URL of physics backend (simplephysics or jaxphysics)"
    )


class OutputSchema(BaseModel):
    """Output schema for integrator tesseract"""
    times: Differentiable[Array[(500,), Float32]] = Field(
        ..., description="Time points of trajectory")
    x_positions: Differentiable[Array[(500,), Float32]] = Field(
        ..., description="X positions along trajectory")
    y_positions: Differentiable[Array[(500,), Float32]] = Field(
        ..., description="Y positions along trajectory")
    z_positions: Differentiable[Array[(500,), Float32]] = Field(
        ..., description="Z positions along trajectory")
    velocities: Differentiable[Array[(500, 3), Float32]] = Field(
        ..., description="Velocity vectors along trajectory")


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the integrator to get ball trajectory"""
    # Run simulation with specified physics backend
    times, x, y, z, velocities = simulate_trajectory(
        initial_velocity=inputs.initial_velocity,
        release_angle=inputs.release_angle,
        roughness=inputs.roughness,
        seam_angle=inputs.seam_angle,
        physics_url=inputs.physics_url
    )

    # Return trajectory data
    return OutputSchema(
        times=times,
        x_positions=x,
        y_positions=y,
        z_positions=z,
        velocities=velocities
    )


def apply_jit(inputs: dict) -> dict:
    """JAX-compatible entry point for AD"""
    times, x, y, z, velocities = simulate_trajectory(
        initial_velocity=inputs["initial_velocity"],
        release_angle=inputs["release_angle"],
        roughness=inputs["roughness"],
        seam_angle=inputs["seam_angle"],
        physics_url=inputs.get("physics_url", "http://simplephysics:8000")
    )
    return {
        "times": times,
        "x_positions": x,
        "y_positions": y,
        "z_positions": z,
        "velocities": velocities
    }


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Abstract evaluation for JAX AD support"""
    max_steps = 500
    return {
        "times": ShapeDType(shape=(max_steps,), dtype="float32"),
        "x_positions": ShapeDType(shape=(max_steps,), dtype="float32"),
        "y_positions": ShapeDType(shape=(max_steps,), dtype="float32"),
        "z_positions": ShapeDType(shape=(max_steps,), dtype="float32"),
        "velocities": ShapeDType(shape=(max_steps, 3), dtype="float32")
    }


#
# JAX handled AD endpoints
#

def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths
    filtered_apply = filter_func(apply_jit, inputs.model_dump(), tuple(jac_outputs))
    return jax.jacrev(filtered_apply)(flatten_with_paths(inputs.model_dump(), include_paths=tuple(jac_inputs)))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths
    filtered_apply = filter_func(apply_jit, inputs.model_dump(), tuple(jvp_outputs))
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs.model_dump(), include_paths=tuple(jvp_inputs))],
        [tangent_vector],
    )[1]


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths
    filtered_apply = filter_func(apply_jit, inputs.model_dump(), tuple(vjp_outputs))
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs.model_dump(), include_paths=tuple(vjp_inputs))
    )
    return vjp_func(cotangent_vector)[0]
