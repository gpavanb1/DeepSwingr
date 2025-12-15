"""
Cricket Ball Trajectory Simulator
Simulates ball flight over 22 yards using trained neural network for aerodynamics.

Usage:
    python cricket_trajectory_sim.py

Requires: matplotlib, jax, plotly, trained_state from the neural network training
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import numpy as np

# Try to import plotly for interactive 3D
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Install plotly for interactive 3D plots: pip install plotly")

# ============================================================================
# PHYSICS SIMULATION
# ============================================================================

def simulate_trajectory(
    trained_state,
    initial_velocity: float,
    release_angle: float,
    roughness: float,
    seam_angle: float,
    dt: float = 0.001,
    pitch_length: float = 20.12,
    debug: bool = False
):
    """Simulate cricket ball trajectory using trained NN for aerodynamic forces."""
    v0 = initial_velocity
    theta = jnp.deg2rad(release_angle)

    vx = v0 * jnp.cos(theta)
    vy = 0.0
    vz = v0 * jnp.sin(theta)

    x, y, z = 0.0, 0.0, 2.0

    mass = 0.156
    diameter = 0.07
    rho_air = 1.225
    mu = 1.5e-5
    g = 9.81

    positions = [[x, y, z]]
    velocities = [[vx, vy, vz]]
    times = [0.0]

    t = 0.0
    max_time = 5.0
    step_count = 0

    while x < pitch_length and z > 0 and t < max_time:
        v_mag = jnp.sqrt(vx**2 + vy**2 + vz**2)
        Re = rho_air * v_mag * diameter / mu
        Re = jnp.clip(Re, 1e5, 1e6)

        forces = trained_state.apply_fn(
            trained_state.params,
            jnp.array([roughness, seam_angle, Re])
        )

        if debug and step_count < 3:
            print(f"\nStep {step_count}:")
            print(f"  Velocity: {v_mag:.2f} m/s ({v_mag*3.6:.1f} km/h)")
            print(f"  Reynolds: {Re:.2e}")
            print(f"  NN Forces: Drag={forces[0]:.4f}N, Lift={forces[1]:.4f}N, Side={forces[2]:.4f}N")

        forces = np.array(forces)

        if v_mag > 1e-6:
            vx_norm = vx / v_mag
            vy_norm = vy / v_mag
            vz_norm = vz / v_mag
        else:
            break

        F_drag_x = -forces[0] * vx_norm
        F_drag_y = -forces[0] * vy_norm
        F_drag_z = -forces[0] * vz_norm

        F_lift_z = forces[1]
        F_swing_y = forces[2]

        Fx = F_drag_x
        Fy = F_drag_y + F_swing_y
        Fz = F_drag_z + F_lift_z - mass * g

        if debug and step_count < 3:
            print(f"  Applied Forces: Fx={Fx:.4f}N, Fy={Fy:.4f}N, Fz={Fz:.4f}N")
            print(f"  Gravity: {-mass*g:.4f}N")

        ax = Fx / mass
        ay = Fy / mass
        az = Fz / mass

        vx = vx + ax * dt
        vy = vy + ay * dt
        vz = vz + az * dt

        x = x + vx * dt
        y = y + vy * dt
        z = z + vz * dt

        t = t + dt
        step_count += 1

        if step_count % 10 == 0:
            positions.append([float(x), float(y), float(z)])
            velocities.append([float(vx), float(vy), float(vz)])
            times.append(float(t))

    if debug:
        print(f"\nFinal statistics:")
        print(f"  Total steps: {step_count}")
        print(f"  Flight time: {t:.3f}s")
        print(f"  Distance: {x:.2f}m")
        print(f"  Lateral deviation: {abs(y)*100:.2f}cm")
        print(f"  Final position: x={x:.2f}, y={y:.4f}, z={z:.2f}")

    positions = np.array(positions)
    velocities = np.array(velocities)
    times = np.array(times)

    if debug:
        print(f"  Y range: min={positions[:, 1].min():.4f}, max={positions[:, 1].max():.4f}")

    return times, positions[:, 0], positions[:, 1], positions[:, 2], velocities


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_trajectory_3d_interactive(x, y, z, initial_velocity, roughness, seam_angle):
    """Create interactive 3D plot using Plotly (works in Jupyter!)"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    # Create trajectory trace
    trajectory = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Ball path',
        hovertemplate='Distance: %{x:.2f}m<br>Lateral: %{y:.2f}m<br>Height: %{z:.2f}m<extra></extra>'
    )

    # Release point
    release = go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers',
        marker=dict(size=8, color='green', symbol='circle'),
        name='Release',
        hovertext=f'Release: {initial_velocity*3.6:.1f} km/h'
    )

    # Landing point
    landing = go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='x'),
        name='Landing',
        hovertext=f'Swing: {abs(y[-1]-y[0])*100:.1f} cm'
    )

    # Pitch outline (ground)
    pitch_x = [0, 20.12, 20.12, 0, 0]
    pitch_y_left = [-1.22, -1.22, -1.22, -1.22, -1.22]
    pitch_y_right = [1.22, 1.22, 1.22, 1.22, 1.22]
    pitch_z = [0, 0, 0, 0, 0]

    pitch_left = go.Scatter3d(
        x=pitch_x, y=pitch_y_left, z=pitch_z,
        mode='lines',
        line=dict(color='brown', width=2),
        showlegend=False,
        hoverinfo='skip'
    )

    pitch_right = go.Scatter3d(
        x=pitch_x, y=pitch_y_right, z=pitch_z,
        mode='lines',
        line=dict(color='brown', width=2),
        showlegend=False,
        hoverinfo='skip'
    )

    # Center line
    center_line = go.Scatter3d(
        x=[0, 20.12], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    )

    # Create figure
    fig = go.Figure(data=[trajectory, release, landing, pitch_left, pitch_right, center_line])

    # Update layout
    fig.update_layout(
        title=f'Interactive 3D Trajectory<br><sub>Roughness: {roughness:.2f}, Seam: {seam_angle:.0f}°, Speed: {initial_velocity*3.6:.0f} km/h</sub>',
        scene=dict(
            xaxis=dict(title='Distance (m)', range=[0, 22]),
            yaxis=dict(title='Lateral (m)', range=[-2, 2]),
            zaxis=dict(title='Height (m)', range=[0, 3]),
            aspectmode='manual',
            aspectratio=dict(x=2, y=0.5, z=0.3),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8)
            )
        ),
        width=900,
        height=600,
        hovermode='closest'
    )

    return fig


def plot_trajectory_3d(t, x, y, z, velocities, initial_velocity, roughness, seam_angle, use_plotly=True):
    """Create 3D visualization of ball trajectory."""

    # Try interactive plotly first if in Jupyter
    if use_plotly and PLOTLY_AVAILABLE:
        fig_interactive = plot_trajectory_3d_interactive(x, y, z, initial_velocity, roughness, seam_angle)
        if fig_interactive is not None:
            fig_interactive.show()

    # Always create matplotlib 2D plots
    fig_2d = plt.figure(figsize=(15, 10))

    # Top view (swing)
    ax2 = fig_2d.add_subplot(2, 2, 1)
    ax2.plot(x, y, 'b-', linewidth=2)
    ax2.scatter(x[0], y[0], c='green', s=100, marker='o')
    ax2.scatter(x[-1], y[-1], c='red', s=100, marker='x')

    # Pitch markings
    pitch_width = 2.44  # 8 feet
    ax2.add_patch(Rectangle((0, -pitch_width/2), 20.12, pitch_width,
                            fill=False, edgecolor='brown', linewidth=2))
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Lateral deviation (m)')
    ax2.set_title('Top View (Swing)')
    ax2.set_xlim(-1, 22)
    ax2.set_ylim(-1, 1)  # Tighter y-axis to see swing better
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Side view (bounce)
    ax3 = fig_2d.add_subplot(2, 2, 2)
    ax3.plot(x, z, 'b-', linewidth=2)
    ax3.scatter(x[0], z[0], c='green', s=100, marker='o')
    ax3.scatter(x[-1], z[-1], c='red', s=100, marker='x')
    ax3.axhline(y=0, color='brown', linewidth=3, label='Pitch')

    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Side View (Flight)')
    ax3.set_xlim(0, 22)
    ax3.set_ylim(0, 3)
    ax3.grid(True, alpha=0.3)

    # Velocity magnitude over time
    ax4 = fig_2d.add_subplot(2, 2, 3)
    v_mag = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2 + velocities[:, 2]**2)
    ax4.plot(t, v_mag * 3.6, 'b-', linewidth=2)  # Convert to km/h
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (km/h)')
    ax4.set_title('Ball Speed')
    ax4.grid(True, alpha=0.3)

    # Lateral displacement over distance
    ax5 = fig_2d.add_subplot(2, 2, 4)
    ax5.plot(x, y * 100, 'r-', linewidth=2)  # Convert to cm
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Distance (m)')
    ax5.set_ylabel('Lateral swing (cm)')
    ax5.set_title('Swing Development')
    ax5.grid(True, alpha=0.3)
    ax5.fill_between(x, 0, y * 100, alpha=0.3, color='red')

    # Add delivery info
    pitch_distance = x[-1]
    lateral_deviation = abs(y[-1] - y[0])
    flight_time = t[-1]
    final_speed = v_mag[-1] * 3.6

    info_text = f"""
Delivery Parameters:
• Initial Speed: {initial_velocity*3.6:.1f} km/h
• Release Angle: {np.rad2deg(np.arcsin(velocities[0, 2]/initial_velocity)):.1f}°
• Roughness: {roughness:.2f}
• Seam Angle: {seam_angle:.1f}°

Results:
• Pitch Distance: {pitch_distance:.2f} m
• Lateral Swing: {lateral_deviation*100:.1f} cm
• Flight Time: {flight_time:.3f} s
• Final Speed: {final_speed:.1f} km/h
• Speed Loss: {(initial_velocity*3.6 - final_speed):.1f} km/h
    """

    fig_2d.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig_2d


def animate_trajectory(t, x, y, z, initial_velocity, roughness, seam_angle):
    """Create animated visualization of ball trajectory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Top view
    ax1.set_xlim(-1, 22)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Lateral deviation (m)')
    ax1.set_title('Top View - Swing')
    ax1.grid(True, alpha=0.3)

    # Pitch
    pitch_width = 2.44
    ax1.add_patch(Rectangle((0, -pitch_width/2), 20.12, pitch_width,
                            fill=False, edgecolor='brown', linewidth=2))
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Side view
    ax2.set_xlim(0, 22)
    ax2.set_ylim(0, 3)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Side View - Flight')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='brown', linewidth=3)

    # Initialize plots
    line1, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.6)
    ball1, = ax1.plot([], [], 'ro', markersize=10)

    line2, = ax2.plot([], [], 'b-', linewidth=2, alpha=0.6)
    ball2, = ax2.plot([], [], 'ro', markersize=10)

    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

    def init():
        line1.set_data([], [])
        ball1.set_data([], [])
        line2.set_data([], [])
        ball2.set_data([], [])
        time_text.set_text('')
        return line1, ball1, line2, ball2, time_text

    def update(frame):
        # Trail
        line1.set_data(x[:frame], y[:frame])
        line2.set_data(x[:frame], z[:frame])

        # Ball position
        if frame < len(x):
            ball1.set_data([x[frame]], [y[frame]])
            ball2.set_data([x[frame]], [z[frame]])
            time_text.set_text(f'Time: {t[frame]:.3f}s\nDistance: {x[frame]:.1f}m')

        return line1, ball1, line2, ball2, time_text

    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(x), interval=20, blit=True, repeat=True)

    plt.tight_layout()
    return fig, anim


# ============================================================================
# INTERACTIVE DEMO
# ============================================================================

def run_interactive_demo(trained_state):
    """Run interactive cricket ball trajectory simulator."""
    print("=" * 70)
    print("CRICKET BALL TRAJECTORY SIMULATOR")
    print("Using Neural Network for Aerodynamics")
    print("=" * 70)

    # Predefined scenarios
    scenarios = {
        '1': {
            'name': 'Outswinger (Fast bowler)',
            'velocity': 44.4,
            'angle': 0,
            'roughness': 0.3,
            'seam_angle': -20,  # Negative angle = outswing (away from batsman, +y)
        },
        '2': {
            'name': 'Inswinger (Medium pace)',
            'velocity': 36.1,
            'angle': 2,
            'roughness': 0.3,
            'seam_angle': 20,  # Positive angle = inswing (toward batsman, -y)
        },
        '3': {
            'name': 'Reverse swing inswinger (Old ball)',
            'velocity': 41.7,
            'angle': 0,
            'roughness': 0.85,
            'seam_angle': -20,  # Negative + high rough = reverse inswing (-y)
        },
        '4': {
            'name': 'Reverse swing outswinger (Old ball)',
            'velocity': 38.9,
            'angle': 1,
            'roughness': 0.85,
            'seam_angle': 25,  # Positive + high rough = reverse outswing (+y)
        },
        '5': {
            'name': 'Custom delivery',
            'velocity': None,
            'angle': None,
            'roughness': None,
            'seam_angle': None,
        }
    }

    print("\nSelect a delivery type:")
    for key, scenario in scenarios.items():
        print(f"  {key}. {scenario['name']}")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice not in scenarios:
        print("Invalid choice, using default (1)")
        choice = '1'

    scenario = scenarios[choice]

    if choice == '5':
        print("\nEnter custom parameters:")
        velocity = float(input("  Initial velocity (km/h, e.g., 140): ")) / 3.6
        angle = float(input("  Release angle (degrees, e.g., 0-5): "))
        roughness = float(input("  Ball roughness (0.0-1.0, e.g., 0.5): "))
        seam_angle = float(input("  Seam angle (-90 to +90, negative=outswing, positive=inswing): "))
    else:
        velocity = scenario['velocity']
        angle = scenario['angle']
        roughness = scenario['roughness']
        seam_angle = scenario['seam_angle']
        print(f"\nSimulating: {scenario['name']}")
        print(f"  Velocity: {velocity*3.6:.1f} km/h")
        print(f"  Angle: {angle}°")
        print(f"  Roughness: {roughness:.2f}")
        print(f"  Seam angle: {seam_angle}°")

    print("\nSimulating trajectory...")
    t, x, y, z, velocities = simulate_trajectory(
        trained_state, velocity, angle, roughness, seam_angle, debug=True
    )

    print(f"✓ Simulation complete! ({len(x)} points)")

    # Debug: print actual y values
    print(f"\nTrajectory check:")
    print(f"  Y-position range: {y.min():.4f} to {y.max():.4f} meters")
    print(f"  Total lateral swing: {abs(y[-1] - y[0])*100:.2f} cm")

    # Create visualizations
    print("\nGenerating visualizations...")
    fig_2d = plot_trajectory_3d(t, x, y, z, velocities, velocity, roughness, seam_angle, use_plotly=True)

    plt.show()

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


# ============================================================================
# BATCH COMPARISON
# ============================================================================

def compare_deliveries(trained_state):
    """Compare multiple delivery types side by side."""

    deliveries = [
        ('Conv. Outswing', 44.4, 0, 0.3, -20),    # Neg angle + low rough = outswing (+y)
        ('Conv. Inswing', 41.7, 0, 0.5, 40),      # Pos angle + low rough = inswing (-y)
        ('Rev. Inswing', 41.7, 0, 0.85, -25),     # Neg angle + high rough = reverse inswing (-y)
        ('Rev. Outswing', 36.1, 2, 0.85, 20),     # Pos angle + high rough = reverse outswing (+y)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (name, vel, ang, rough, seam) in enumerate(deliveries):
        t, x, y, z, _ = simulate_trajectory(trained_state, vel, ang, rough, seam)

        ax = axes[idx]
        ax.plot(x, y, linewidth=2, label='Trajectory')
        ax.scatter(x[0], y[0], c='green', s=100, marker='o', label='Release')
        ax.scatter(x[-1], y[-1], c='red', s=100, marker='x', label='Landing')

        # Pitch
        pitch_width = 2.44
        ax.add_patch(Rectangle((0, -pitch_width/2), 20.12, pitch_width,
                               fill=False, edgecolor='brown', linewidth=2))
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Lateral deviation (m)')
        ax.set_title(f'{name}\n{vel*3.6:.0f} km/h, Rough={rough:.1f}, Seam={seam}°')
        ax.set_xlim(-1, 22)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add swing measurement
        swing = abs(y[-1] - y[0]) * 100
        ax.text(0.95, 0.05, f'Swing: {swing:.1f} cm',
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_interactive_demo(trained_state)
    print("\nNOTE: Import your trained_state from the training script first!")
    print("Example:")
    print("  from cricket_training import train_model_with_embedded_cfd")
    print("  trained_state = train_model_with_embedded_cfd(...)")
    print("  run_interactive_demo(trained_state)")
    print("\nOr use: compare_deliveries(trained_state)")
