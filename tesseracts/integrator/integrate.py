"""
Cricket Ball Trajectory Simulator
Simulates ball flight over 22 yards using configurable physics backend.
Uses Diffrax for JAX-native differentiable integration.
"""
import jax
import jax.numpy as jnp
import diffrax as dfx
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract


def simulate_trajectory(
    initial_velocity: float,
    release_angle: float,
    roughness: float,
    seam_angle: float,
    physics_url: str = "http://simplephysics:8000",
    dt: float = 0.08, # Increased default dt for faster simulation
    pitch_length: float = 20.12,
    debug: bool = False
):
    """
    Simulate cricket ball trajectory using Diffrax and configurable physics.
    """
    # Constants
    mass = 0.156  # kg
    diameter = 0.07  # m
    rho_air = 1.225  # kg/m³
    mu = 1.5e-5  # Pa·s
    g = 9.81  # m/s²

    # Connect to physics backend
    physics = Tesseract.from_url(physics_url)

    def ball_dynamics(t, y, args):
        """System of ODEs for ball motion."""
        x, y_pos, z, vx, vy, vz = y

        # Calculate velocity magnitude and Reynolds number
        v_mag = jnp.sqrt(vx**2 + vy**2 + vz**2)
        Re = rho_air * v_mag * diameter / mu
        Re = jnp.clip(Re, 1e5, 1e6)

        # Get forces from physics tesseract differentiably
        forces_res = apply_tesseract(physics, {
            "notch_angle": seam_angle,
            "reynolds_number": Re,
            "roughness": roughness
        })
        forces = forces_res['force_vector']

        # Unit vectors
        v_safe = v_mag + 1e-6
        vx_norm = vx / v_safe
        vy_norm = vy / v_safe
        vz_norm = vz / v_safe

        # Force components: forces[0]=drag, forces[1]=lift, forces[2]=swing
        F_drag_x = -forces[0] * vx_norm
        F_drag_y = -forces[0] * vy_norm
        F_drag_z = -forces[0] * vz_norm

        F_lift_z = forces[1]
        F_swing_y = forces[2]

        # Total forces
        Fx = F_drag_x
        Fy = F_drag_y + F_swing_y
        Fz = F_drag_z + F_lift_z - mass * g

        # Accelerations
        ax = Fx / mass
        ay = Fy / mass
        az = Fz / mass

        # Use a soft switch for ground/pitch end to keep gradients flowing
        active = jnp.logical_and(z > 0.0, x < pitch_length)
        
        return jnp.where(
            active,
            jnp.array([vx, vy, vz, ax, ay, az]),
            jnp.zeros(6)
        )

    # Initial conditions
    theta = jnp.deg2rad(release_angle)
    v0_x = initial_velocity * jnp.cos(theta)
    v0_y = 0.0
    v0_z = initial_velocity * jnp.sin(theta)

    y0 = jnp.array([0.0, 0.0, 2.0, v0_x, v0_y, v0_z])

    # Solve ODE system
    term = dfx.ODETerm(ball_dynamics)
    
    # Use Heun's method (2nd order) for faster execution
    solver = dfx.Heun()
    
    # We want exactly 500 points for the trajectory
    max_steps = 500
    t0, t1 = 0.0, 1.2  # Ball always finishes before 1.2s
    
    # Use SaveAt to get exactly 500 points
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, max_steps))
    
    # Use RecursiveCheckpointAdjoint for MUCH faster gradients
    # and ConstantStepSize to avoid adaptive overhead
    @jax.jit
    def run_solve(y0_val):
        return dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=y0_val,
            saveat=saveat,
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            max_steps=max_steps,
        )

    sol = run_solve(y0)

    times = sol.ts
    x = sol.ys[:, 0]
    y = sol.ys[:, 1]
    z = sol.ys[:, 2]
    vx = sol.ys[:, 3]
    vy = sol.ys[:, 4]
    vz = sol.ys[:, 5]
    
    velocities = jnp.stack([vx, vy, vz], axis=-1)

    return times, x, y, z, velocities
