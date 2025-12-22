# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tesseract API module for optimizer tesseract
Inputs: fixed_variables, optimization_variables, swing_type, swing_url, physics_url, integrator_url
Outputs: optimal_parameters, maximum_deviation
"""
from typing import Any, Dict, List, Literal
import numpy as np
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field, model_validator

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
    """Optimize swing parameters using distributed AD and L-BFGS-B"""
    # Connect to swing tesseract
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract
    swing_tesseract = Tesseract.from_url(inputs.swing_url)

    # Get optimization variable names and bounds
    opt_vars = list(inputs.optimization_variables.keys())
    
    # Create objective function with distributed Jacobian
    def objective_with_grad(x):
        # Convert to JAX array for AD
        x_jax = jnp.array(x)

        def jax_objective(x_jax):
            params_dict = {opt_vars[i]: x_jax[i] for i in range(len(opt_vars))}

            # Prepare input for swing tesseract
            # We pass the JAX tracers directly into the framework
            swing_inputs = {
                "initial_velocity": inputs.fixed_variables.get("initial_velocity", 35.0),
                "release_angle": inputs.fixed_variables.get("release_angle", 5.0),
                "roughness": inputs.fixed_variables.get("roughness", 0.8),
                "seam_angle": params_dict.get("seam_angle", 30.0)
            }

            # Add any other optimization variables
            for var, val in params_dict.items():
                swing_inputs[var] = val

            # Use apply_tesseract from tesseract-jax for AD compatibility.
            # This ensures that when JAX calls this within a transform (like jax.grad),
            # it correctly handles tracers and makes the appropriate distributed calls.
            res = apply_tesseract(swing_tesseract, swing_inputs)
            deviation = res["final_deviation"]

            # Logic for objective based on swing_type
            if inputs.swing_type == "in":
                score = deviation  # Minimize deviation (towards batsman)
            else: # out
                score = -deviation  # Minimize -deviation (away from batsman)

            return score

        # Compute function value and gradient using distributed AD
        # The Tesseract SDK automatically handles the network calls to .jacobian
        # when jax.grad is used on a function calling tesseract.apply()
        score = jax_objective(x_jax)
        grad_jax = jax.grad(jax_objective)(x_jax)

        # Convert back to numpy for scipy optimizer
        grad = np.array(grad_jax)
        
        # Convert score back to deviation for display
        deviation = -float(score) if inputs.swing_type == "out" else float(score)
        params_dict = {opt_vars[i]: float(x[i]) for i in range(len(opt_vars))}

        print(f"  Evaluation: deviation={deviation:.2f} cm, score={float(score):.2f}, params={params_dict}")
        return float(score), grad.astype(float)

    # Use L-BFGS-B for bounded optimization with gradients
    try:
        import time
        from scipy.optimize import minimize
        start_time = time.time()
        print(f"Starting L-BFGS-B optimization using distributed AD Jacobian...")

        # Initial guess: middle of each bound
        x0 = []
        bounds = []
        for var in opt_vars:
            b = inputs.optimization_variables[var]
            x0.append((b[0] + b[1]) / 2.0)
            bounds.append((b[0], b[1]))

        # Run optimization
        result = minimize(
            objective_with_grad,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 20,
                'ftol': 1e-3,
            }
        )

        elapsed = time.time() - start_time
        print(f"Optimization completed in {elapsed:.1f} seconds")
        print(f"Success: {result.success}, Evaluations: {result.nfev}")

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Extract optimal parameters
    optimal_params = {opt_vars[i]: float(result.x[i]) for i in range(len(opt_vars))}
    
    # Get final deviation for the output
    final_res = swing_tesseract.apply({
        **inputs.fixed_variables,
        **optimal_params
    })
    final_deviation = float(final_res["final_deviation"])

    return OutputSchema(
        optimal_parameters=optimal_params,
        maximum_deviation=abs(final_deviation)
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the optimizer to find optimal swing parameters"""
    print(f"Optimizer received request: optimizing {list(inputs.optimization_variables.keys())}")
    return optimize_swing(inputs)
