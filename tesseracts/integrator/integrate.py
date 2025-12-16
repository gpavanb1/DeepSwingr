"""
Cricket Ball Trajectory Simulator
Simulates ball flight over 22 yards using configurable physics backend.
"""
import numpy as np
from tesseract_core import Tesseract
from scipy.integrate import solve_ivp


def simulate_trajectory(
    initial_velocity: float,
    release_angle: float,
    roughness: float,
    seam_angle: float,
    physics_url: str = "http://simplephysics:8000",
    dt: float = 0.001,
    pitch_length: float = 20.12,
    debug: bool = False
):
    """
    Simulate cricket ball trajectory using configurable physics backend.

    Args:
        initial_velocity: Initial ball velocity in m/s
        release_angle: Release angle in degrees
        roughness: Surface roughness coefficient [0.0, 1.0]
        seam_angle: Seam angle in degrees [-90, 90]
        physics_url: URL of physics tesseract (simplephysics or jaxphysics)
        dt: Time step for integration
        pitch_length: Length of cricket pitch in meters
        debug: Print debug information

    Returns:
        times, x_positions, y_positions, z_positions, velocities
    """
    # Constants
    mass = 0.156  # kg
    diameter = 0.07  # m
    rho_air = 1.225  # kg/m³
    mu = 1.5e-5  # Pa·s
    g = 9.81  # m/s²

    # Initial conditions
    theta = np.deg2rad(release_angle)
    v0_x = initial_velocity * np.cos(theta)
    v0_y = 0.0
    v0_z = initial_velocity * np.sin(theta)

    y0 = [0.0, 0.0, 2.0,  # Initial positions (x, y, z)
          v0_x, v0_y, v0_z]  # Initial velocities (vx, vy, vz)

    # Connect to physics backend via URL
    physics = Tesseract.from_url(physics_url)

    if debug:
        print(f"Using physics backend: {physics_url}")

    def ball_dynamics(t, y):
        """System of ODEs for ball motion."""
        x, y_pos, z, vx, vy, vz = y

        # Calculate velocity magnitude and Reynolds number
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        Re = rho_air * v_mag * diameter / mu
        Re = np.clip(Re, 1e5, 1e6)

        if v_mag < 1e-6:
            return [0, 0, 0, 0, 0, 0]

        # Get forces from physics tesseract
        forces = physics.apply({
            "notch_angle": seam_angle,
            "reynolds_number": float(Re),
            "roughness": roughness
        })['force_vector']

        # Unit vectors for velocity components
        vx_norm = vx / v_mag
        vy_norm = vy / v_mag
        vz_norm = vz / v_mag

        # Force components
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

        return [vx, vy, vz, ax, ay, az]

    def event_ground_or_pitch(t, y):
        """Stop integration when ball hits ground or reaches pitch length."""
        return min(y[2], pitch_length - y[0])  # z=0 or x=pitch_length

    event_ground_or_pitch.terminal = True

    # Solve ODE system
    t_span = (0, 5.0)  # Max time 5 seconds
    t_eval = np.arange(0, 5.0, dt)

    solution = solve_ivp(
        ball_dynamics,
        t_span,
        y0,
        method='RK45',
        t_eval=t_eval,
        events=event_ground_or_pitch,
        rtol=1e-6,
        atol=1e-6
    )

    # Extract results
    times = solution.t
    positions = np.column_stack([
        solution.y[0],  # x positions
        solution.y[1],  # y positions
        solution.y[2]   # z positions
    ])

    velocities = np.column_stack([
        solution.y[3],  # vx
        solution.y[4],  # vy
        solution.y[5]   # vz
    ])

    if debug:
        physics_name = "JAX-CFD" if "jaxphysics" in physics_url else "Simple"
        print(f"\n[{physics_name}] Final statistics:")
        print(f"  Total steps: {len(times)}")
        print(f"  Flight time: {times[-1]:.3f}s")
        print(f"  Distance: {positions[-1, 0]:.2f}m")
        print(f"  Lateral deviation: {abs(positions[-1, 1])*100:.2f}cm")
        print(
            f"  Final position: x={positions[-1, 0]:.2f}, y={positions[-1, 1]:.4f}, z={positions[-1, 2]:.2f}")
        print(
            f"  Y range: min={positions[:, 1].min():.4f}, max={positions[:, 1].max():.4f}")

    return times, positions[:, 0], positions[:, 1], positions[:, 2], velocities
