# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tesseract API module for optimizer tesseract
Inputs: fixed_variables, optimization_variables, swing_type, swing_url, physics_url, integrator_url
Outputs: optimal_parameters, maximum_deviation
"""
from typing import Dict, List, Literal
import numpy as np
import time
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field, model_validator
from scipy.optimize import minimize_scalar, minimize

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
    use_jacobian: bool = Field(
        default=False,
        description="Whether to use Jacobian gradients for optimization")

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
    """Optimize swing parameters"""
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract
    swing_tesseract = Tesseract.from_url(inputs.swing_url)

    # Get optimization variable names and bounds
    opt_vars = list(inputs.optimization_variables.keys())
    opt_var = opt_vars[0] if opt_vars else "seam_angle"
    
    # Pre-calculate bounds for both optimization paths
    bounds_list = []
    x0_list = []
    for var in opt_vars:
        b = inputs.optimization_variables[var]
        bounds_list.append((float(b[0]), float(b[1])))
        x0_list.append(float(b[0] + b[1]) / 2.0)
    
    # For 1D optimization (Brent)
    current_bounds = inputs.optimization_variables[opt_var]

    if inputs.use_jacobian:
        print(f"Starting L-BFGS-B optimization using distributed AD Jacobian for {opt_vars}...")
        
        def objective_with_grad(x):
            # Convert to JAX array for AD
            x_jax = jnp.array(x)

            def jax_objective(x_jax):
                params_dict = {opt_vars[i]: x_jax[i] for i in range(len(opt_vars))}

                # Prepare input for swing tesseract
                swing_inputs = {
                    **inputs.fixed_variables,
                    "physics_url": inputs.physics_url
                }

                # Overwrite optimization variables with tracers
                for var, val in params_dict.items():
                    swing_inputs[var] = val

                # Distributed AD call via tesseract-jax
                res = apply_tesseract(swing_tesseract, swing_inputs)
                deviation = res["final_deviation"]

                return deviation if inputs.swing_type == "in" else -deviation

            # Compute function value and gradient
            score, grad_jax = jax.value_and_grad(jax_objective)(x_jax)
            
            grad = np.array(grad_jax).astype(float)
            deviation = -float(score) if inputs.swing_type == "out" else float(score)
            
            print(f"  Check params={x} -> deviation={deviation:.2f} cm")
            return float(score), grad

        try:
            start_time = time.time()
            result = minimize(
                objective_with_grad,
                x0=x0_list,
                bounds=bounds_list,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 20, 'ftol': 1e-3}
            )
            elapsed = time.time() - start_time
            print(f"Optimization completed in {elapsed:.1f} seconds ({result.nfev} evaluations)")
            
            optimal_params = {opt_vars[i]: float(result.x[i]) for i in range(len(opt_vars))}
            max_dev = -float(result.fun) if inputs.swing_type == "out" else float(result.fun)
            
        except Exception as e:
            print(f"L-BFGS-B Optimization failed: {e}")
            raise

    else:
        def objective(x):
            # Prepare input for swing tesseract
            swing_inputs = {
                **inputs.fixed_variables,
                opt_var: float(x),
                "physics_url": inputs.physics_url
            }

            # Just a forward pass
            res = swing_tesseract.apply(swing_inputs)
            deviation = float(res["final_deviation"])

            # Minimize negative deviation for "out", positive for "in"
            score = deviation if inputs.swing_type == "in" else -deviation
            
            print(f"  Check {opt_var}={x:.2f} -> deviation={deviation:.2f} cm")
            return score

        try:
            start_time = time.time()
            print(f"Starting fast 1D search for optimal {opt_var}...")

            # Brent's method is extremely fast for 1D smooth functions
            result = minimize_scalar(
                objective,
                bounds=(current_bounds[0], current_bounds[1]),
                method='bounded',
                options={'xatol': 0.1, 'maxiter': 10}
            )

            elapsed = time.time() - start_time
            print(f"Optimization completed in {elapsed:.1f} seconds ({result.nfev} evaluations)")
            
            optimal_params = {opt_var: float(result.x)}
            max_dev = -float(result.fun) if inputs.swing_type == "out" else float(result.fun)

        except Exception as e:
            print(f"Brent Optimization failed: {e}")
            raise

    return OutputSchema(
        optimal_parameters=optimal_params,
        maximum_deviation=abs(max_dev)
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the optimizer to find optimal swing parameters"""
    print(f"Optimizer received request: optimizing {list(inputs.optimization_variables.keys())}")
    return optimize_swing(inputs)
