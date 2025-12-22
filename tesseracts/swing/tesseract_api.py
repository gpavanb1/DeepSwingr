import jax
import jax.numpy as jnp
from typing import Any
from pydantic import BaseModel, Field
from tesseract_jax import apply_tesseract
from tesseract_core import Tesseract

from tesseract_core.runtime import Float32, Differentiable, ShapeDType


class InputSchema(BaseModel):
    """Input schema for swing tesseract"""
    initial_velocity: Differentiable[Float32] = Field(
        ..., description="Initial velocity in m/s [20.0, 50.0]")
    release_angle: Differentiable[Float32] = Field(
        ..., description="Release angle in degrees [0.0, 10.0]")
    roughness: Differentiable[Float32] = Field(
        ..., description="Ball surface roughness [0.0, 1.0]")
    seam_angle: Differentiable[Float32] = Field(
        ..., description="Seam angle in degrees [-90.0, 90.0]")


class OutputSchema(BaseModel):
    """Output schema for swing tesseract"""

    final_deviation: Differentiable[Float32] = Field(
        ..., description="Final lateral deviation from swing in cm"
    )


# Connect to integrator tesseract (Docker network alias)
integrator_tesseract = Tesseract.from_url("http://integrator:8000")


@jax.jit
def compute_final_deviation(
    initial_velocity,
    release_angle,
    roughness,
    seam_angle
):
    """
    Compute final lateral deviation using AD-enabled tesseract calls.
    """
    # Use apply_tesseract for AD compatibility through the framework
    # This automatically handles distributed automatic differentiation
    trajectory_result = apply_tesseract(integrator_tesseract, {
        "initial_velocity": initial_velocity,
        "release_angle": release_angle,
        "roughness": roughness,
        "seam_angle": seam_angle,
        "physics_url": "http://simplephysics:8000"
    })

    # Extract y positions (lateral deviation)
    y_positions = trajectory_result['y_positions']

    # Get final lateral position (y component at the end of the pitch)
    final_deviation = y_positions[-1] * 100  # Convert to cm

    return final_deviation


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the swing tesseract to get final deviation from trajectory"""
    # Get trajectory from integrator and compute final deviation
    final_deviation = compute_final_deviation(
        initial_velocity=inputs.initial_velocity,
        release_angle=inputs.release_angle,
        roughness=inputs.roughness,
        seam_angle=inputs.seam_angle
    )

    # Return final deviation
    return OutputSchema(final_deviation=final_deviation)


def apply_jit(inputs: dict) -> dict:
    """JAX-compatible entry point for AD"""
    # Extract inputs directly
    initial_velocity = inputs["initial_velocity"]
    release_angle = inputs["release_angle"]
    roughness = inputs["roughness"]
    seam_angle = inputs["seam_angle"]

    # Call the computation function directly
    final_deviation = compute_final_deviation(
        initial_velocity=initial_velocity,
        release_angle=release_angle,
        roughness=roughness,
        seam_angle=seam_angle
    )

    return {"final_deviation": final_deviation}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Abstract evaluation for JAX AD support"""
    return {"final_deviation": ShapeDType(shape=(), dtype="float32")}


#
# JAX handled AD endpoints (no changes needed)
#

def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    # Imports locally to avoid import issues if needed
    from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

    filtered_apply = filter_func(
        apply_jit, inputs.model_dump(), tuple(jac_outputs))
    return jax.jacrev(filtered_apply)(flatten_with_paths(inputs.model_dump(), include_paths=tuple(jac_inputs)))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

    filtered_apply = filter_func(
        apply_jit, inputs.model_dump(), tuple(jvp_outputs))
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs.model_dump(),
                            include_paths=tuple(jvp_inputs))],
        [tangent_vector],
    )[1]


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

    filtered_apply = filter_func(
        apply_jit, inputs.model_dump(), tuple(vjp_outputs))
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(
            inputs.model_dump(), include_paths=tuple(vjp_inputs))
    )
    return vjp_func(cotangent_vector)[0]
