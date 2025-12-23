# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tesseract API module for optimizer tesseract
Inputs: fixed_variables, optimization_variables, swing_type, swing_url, physics_url, integrator_url
Outputs: optimal_parameters, maximum_deviation
"""
from typing import Any, Dict, List, Literal
import numpy as np
import time
from pydantic import BaseModel, Field, model_validator
from scipy.optimize import minimize_scalar

from tesseract_core.runtime import Float32


class InputSchema(BaseModel):
    """Input schema for optimizer tesseract"""

    fixed_variables: Dict[str, Float32] = Field(
        ..., description="Fixed parameters that don't change during optimization")
    optimization_variables: Dict[str, List[Float32]] = Field(
        ..., description="Variables to optimize with [min, max] bounds")
    swing_type: Literal["in", "out"] = Field(
        ..., description="Type of swing to optimize for")
    swing_url: str = Field(
        default="http://swing:8000",
        description="URL of swing tesseract")
    physics_url: str = Field(
        default="http://simplephysics:8000",
        description="URL of physics backend")
    integrator_url: str = Field(
        default="http://integrator:8000",
        description="URL of integrator tesseract")

    @model_validator(mode="after")
    def validate_inputs(self):
        # Validate that optimization variables have valid bounds
        for var_name, bounds in self.optimization_variables.items():
            if len(bounds) != 2:
                raise ValueError(f"optimization_variables[{var_name}] must have exactly 2 bounds [min, max]")
            if bounds[0] >= bounds[1]:
                raise ValueError(f"optimization_variables[{var_name}] min ({bounds[0]}) must be < max ({bounds[1]})")

        # Validate that swing_type is valid
        if self.swing_type not in ["in", "out"]:
            raise ValueError(f"swing_type must be 'in' or 'out'. Got {self.swing_type}")

        return self


class OutputSchema(BaseModel):
    """Output schema for optimizer tesseract"""

    optimal_parameters: Dict[str, Float32] = Field(
        ..., description="Optimal values for the optimization variables")
    maximum_deviation: Float32 = Field(
        ..., description="Maximum swing deviation achieved (cm)")


def optimize_swing(inputs: InputSchema) -> OutputSchema:
    """Optimize swing parameters using fast scalar search (no gradients)"""
    from tesseract_core import Tesseract
    swing_tesseract = Tesseract.from_url(inputs.swing_url)

    # We assume 1D optimization for seam_angle as it's the primary physical variable
    opt_var = "seam_angle"
    if opt_var not in inputs.optimization_variables:
        # Fallback if seam_angle isn't the target, pick the first available
        opt_var = list(inputs.optimization_variables.keys())[0]
        
    bounds = inputs.optimization_variables[opt_var]
    
    def objective(x):
        # Prepare input for swing tesseract
        swing_inputs = {
            **inputs.fixed_variables,
            opt_var: float(x)
        }

        # Just a forward pass - much faster than a .jacobian() call
        res = swing_tesseract.apply(swing_inputs)
        deviation = float(res["final_deviation"])

        # Minimize negative deviation for "out", positive for "in"
        # We use Brent's method which finds a minimum
        score = deviation if inputs.swing_type == "in" else -deviation
        
        print(f"  Check {opt_var}={x:.2f} -> deviation={deviation:.2f} cm")
        return score

    try:
        start_time = time.time()
        print(f"Starting fast 1D search for optimal {opt_var}...")

        # Brent's method is extremely fast for 1D smooth functions
        result = minimize_scalar(
            objective,
            bounds=(bounds[0], bounds[1]),
            method='bounded',
            options={'xatol': 0.1, 'maxiter': 10} # 10 steps is plenty for 1D
        )

        elapsed = time.time() - start_time
        print(f"Optimization completed in {elapsed:.1f} seconds ({result.nfev} evaluations)")

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    optimal_params = {opt_var: float(result.x)}
    
    # Final deviation is the value found by the optimizer (remembering to flip sign for 'out')
    max_dev = -float(result.fun) if inputs.swing_type == "out" else float(result.fun)

    return OutputSchema(
        optimal_parameters=optimal_params,
        maximum_deviation=abs(max_dev)
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the optimizer to find optimal swing parameters"""
    print(f"Optimizer received request: optimizing {list(inputs.optimization_variables.keys())}")
    return optimize_swing(inputs)
