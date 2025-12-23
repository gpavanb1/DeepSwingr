# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tesseract API module for jaxphysics tesseract
Inputs: notch_angle, reynolds_number, roughness
Outputs: force vector [Fx_drag, Fy_lift, Fz_side]
"""
# Import the physics solver function from physics.py
from typing import Any
import os
import pathlib
import flax
import equinox as eqx
import flax.serialization
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType

# Define path to weights file relative to this file location
this_dir = pathlib.Path(__file__).parent
weights_path = this_dir / "weights" / "jaxphysics_weights.msgpack"


class InputSchema(BaseModel):
    """Input schema for jaxphysics tesseract"""

    notch_angle: Differentiable[Float32] = Field(
        ..., description="Seam angle in degrees [-90, 90]")
    reynolds_number: Differentiable[Float32] = Field(
        ..., description="Flow Reynolds number [1e5, 1e6]")
    roughness: Differentiable[Float32] = Field(
        ..., description="Surface roughness coefficient [0.0, 1.0]")

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        # Only validate if these are not JAX tracers or ShapeDType objects
        # During abstract_eval, these might be ShapeDType or tracers
        def is_concrete(val):
            return not hasattr(val, "shape") and not hasattr(val, "dtype")

        if is_concrete(self.notch_angle):
            if not (-90.0 <= self.notch_angle <= 90.0):
                raise ValueError(
                    f"notch_angle must be in [-90, 90]. Got {self.notch_angle}")
        if is_concrete(self.reynolds_number):
            if not (1e5 <= self.reynolds_number <= 1e6):
                raise ValueError(
                    f"reynolds_number must be in [1e5, 1e6]. Got {self.reynolds_number}")
        if is_concrete(self.roughness):
            if not (0.0 <= self.roughness <= 1.0):
                raise ValueError(
                    f"roughness must be in [0.0, 1.0]. Got {self.roughness}")
        return self


class OutputSchema(BaseModel):
    """Output schema for jaxphysics tesseract"""

    force_vector: Differentiable[Array[(3,), Float32]] = Field(
        ..., description="Force vector [Fx_drag, Fy_lift, Fz_side] in Newtons"
    )


class JaxPhysicsModel(eqx.Module):
    """Encapsulate the neural network and parameters for inference."""

    model: eqx.Module
    params: flax.core.FrozenDict

    def __call__(self, notch_angle, reynolds_number, roughness):
        inputs = jnp.array([roughness, notch_angle, reynolds_number])
        return self.model.apply(self.params, inputs)


def load_weights():
    from physics import CricketBallForceNetwork

    # Create model instance with dummy inputs for shape
    model = CricketBallForceNetwork()

    # Load the saved weights file
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    with open(weights_path, "rb") as f:
        bytes_data = f.read()

    params = flax.serialization.from_bytes(model.init(
        jax.random.PRNGKey(0), jnp.ones(3)), bytes_data)

    return JaxPhysicsModel(model=model, params=params)


# Load weights once on import
model_instance = load_weights()


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    notch_angle = jnp.array(inputs["notch_angle"])
    reynolds_number = jnp.array(inputs["reynolds_number"])
    roughness = jnp.array(inputs["roughness"])

    forces = model_instance(notch_angle, reynolds_number, roughness)

    return {"force_vector": forces}


def apply(inputs: InputSchema) -> OutputSchema:
    # Convert inputs to dict and call jit-compiled apply function
    out = apply_jit(inputs.model_dump())

    # Return OutputSchema object
    return OutputSchema(**out)


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Abstract evaluation for JAX AD support"""
    return {"force_vector": ShapeDType(shape=(3,), dtype="float32")}


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
