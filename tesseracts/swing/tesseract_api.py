# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tesseract API module for swing tesseract
Inputs: initial_velocity, release_angle, roughness, seam_angle, physics_url, integrator_url
Outputs: maximum_deviation (maximum lateral deviation from swing in cm)
"""
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Float32, Differentiable


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
    physics_url: str = Field(
        default="http://simplephysics:8000",
        description="URL of physics backend (simplephysics or jaxphysics)"
    )
    integrator_url: str = Field(
        default="http://integrator:8000",
        description="URL of integrator tesseract"
    )


class OutputSchema(BaseModel):
    """Output schema for swing tesseract"""

    maximum_deviation: Differentiable[Float32] = Field(
        ..., description="Maximum lateral deviation from swing in cm"
    )


def compute_maximum_deviation(
    initial_velocity,
    release_angle,
    roughness,
    seam_angle,
    physics_url: str = "http://simplephysics:8000",
    integrator_url: str = "http://integrator:8000"
):
    """
    Get trajectory from integrator and compute maximum lateral deviation.

    Args:
        initial_velocity: Initial ball velocity in m/s (can be differentiable)
        release_angle: Release angle in degrees (can be differentiable)
        roughness: Surface roughness coefficient [0.0, 1.0] (can be differentiable)
        seam_angle: Seam angle in degrees [-90, 90] (can be differentiable)
        physics_url: URL of physics backend (string)
        integrator_url: URL of integrator tesseract (string)

    Returns:
        Maximum lateral deviation in cm
    """
    # Connect to integrator tesseract
    from tesseract_core import Tesseract
    integrator = Tesseract.from_url(integrator_url)

    # Convert differentiable inputs to regular values for integrator
    # The integrator expects regular float values, not differentiable ones
    vel_value = float(initial_velocity)
    angle_value = float(release_angle)
    rough_value = float(roughness)
    seam_value = float(seam_angle)

    # Get trajectory data from integrator
    trajectory_result = integrator.apply({
        "initial_velocity": vel_value,
        "release_angle": angle_value,
        "roughness": rough_value,
        "seam_angle": seam_value,
        "physics_url": physics_url
    })

    # Extract y positions (lateral deviation)
    y_positions = np.array(trajectory_result['y_positions'])

    # Calculate maximum absolute deviation from initial y=0 position
    max_deviation = np.max(np.abs(y_positions)) * 100  # Convert to cm

    return max_deviation


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the swing tesseract to get maximum deviation from trajectory"""
    # Get trajectory from integrator and compute maximum deviation
    max_deviation = compute_maximum_deviation(
        initial_velocity=inputs.initial_velocity,
        release_angle=inputs.release_angle,
        roughness=inputs.roughness,
        seam_angle=inputs.seam_angle,
        physics_url=inputs.physics_url,
        integrator_url=inputs.integrator_url
    )

    # Return maximum deviation
    return OutputSchema(maximum_deviation=max_deviation)


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
