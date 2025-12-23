import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np

# Try to import plotly for interactive 3D
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Install plotly for interactive 3D plots: pip install plotly")


def compare_deliveries(deliveries):
    """
    Show exactly 4 subplots (one per delivery),
    superimposing Simplephysics and Jaxphysics trajectories.
    """

    # ---- Group by delivery name (strip backend suffix) ----
    grouped = {}
    for name, vel, ang, rough, seam, t, x, y, z in deliveries:
        if "(Simple)" in name:
            base = name.replace(" (Simple)", "")
            backend = "Simple"
        elif "(JAX" in name:
            base = name.replace(" (JAX-CFD)", "")
            backend = "Jax"
        else:
            base = name
            backend = "Unknown"

        grouped.setdefault(base, {
            "velocity": vel,
            "angle": ang,
            "roughness": rough,
            "seam": seam,
            "Simple": None,
            "Jax": None
        })

        grouped[base][backend] = (t, x, y, z)

    delivery_names = list(grouped.keys())[:4]  # hard cap at 4

    # ---- Fixed 2x2 layout ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, dname in zip(axes, delivery_names):
        data = grouped[dname]

        # Pitch
        pitch_width = 2.44
        ax.add_patch(Rectangle(
            (0, -pitch_width / 2), 20.12, pitch_width,
            fill=False, edgecolor='brown', linewidth=2
        ))
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)

        # ---- Plot Simplephysics ----
        if data["Simple"] is not None:
            _, x, y, _ = data["Simple"]
            ax.plot(x, y, color="blue", linewidth=2, label="Simplephysics")
            ax.scatter(x[-1], y[-1], color="blue", marker="x", s=80)

        # ---- Plot Jaxphysics ----
        if data["Jax"] is not None:
            _, x, y, _ = data["Jax"]
            ax.plot(x, y, color="red", linewidth=2,
                    linestyle="--", label="Jaxphysics")
            ax.scatter(x[-1], y[-1], color="red", marker="x", s=80)

        ax.set_title(
            f"{dname}\n"
            f"{data['velocity']*3.6:.0f} km/h | "
            f"Rough={data['roughness']:.2f} | Seam={data['seam']}°"
        )

        ax.set_xlim(-1, 22)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Lateral deviation (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # ---- Swing annotation (if both exist, show both) ----
        text_lines = []
        if data["Simple"] is not None:
            _, x, y, _ = data["Simple"]
            text_lines.append(f"S: {abs(y[-1]-y[0])*100:.1f} cm")
        if data["Jax"] is not None:
            _, x, y, _ = data["Jax"]
            text_lines.append(f"J: {abs(y[-1]-y[0])*100:.1f} cm")

        ax.text(
            0.95, 0.05, "\n".join(text_lines),
            transform=ax.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7)
        )

    # Hide unused axes if <4 deliveries
    for i in range(len(delivery_names), 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


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
            time_text.set_text(
                f'Time: {t[frame]:.3f}s\nDistance: {x[frame]:.1f}m')

        return line1, ball1, line2, ball2, time_text

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(x), interval=20, blit=True, repeat=True)

    plt.tight_layout()
    return fig, anim


def plot_trajectory_3d(t, x, y, z, velocities, initial_velocity, roughness, seam_angle,
                       use_plotly=True, all_trajectories=None):
    """Create 3D visualization of ball trajectory with optional comparison."""

    # If multiple trajectories provided, plot them together
    if use_plotly and PLOTLY_AVAILABLE:
        if all_trajectories is not None and len(all_trajectories) > 1:
            fig_interactive = plot_trajectory_3d_interactive_multi(
                all_trajectories, initial_velocity, roughness, seam_angle)
        else:
            fig_interactive = plot_trajectory_3d_interactive(
                x, y, z, initial_velocity, roughness, seam_angle)
        if fig_interactive is not None:
            return fig_interactive

    # Fallback to matplotlib 2D plots
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
    ax2.set_ylim(-1, 1)
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
    v_mag = np.sqrt(velocities[:, 0]**2 +
                    velocities[:, 1]**2 + velocities[:, 2]**2)
    ax4.plot(t, v_mag * 3.6, 'b-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (km/h)')
    ax4.set_title('Ball Speed')
    ax4.grid(True, alpha=0.3)

    # Lateral displacement over distance
    ax5 = fig_2d.add_subplot(2, 2, 4)
    ax5.plot(x, y * 100, 'r-', linewidth=2)
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
• Roughness: {roughness:.2f}
• Seam Angle: {seam_angle:.1f}°

Results:
• Pitch Distance: {pitch_distance:.2f} m
• Lateral Swing: {lateral_deviation*100:.1f} cm
• Flight Time: {flight_time:.3f} s
• Final Speed: {final_speed:.1f} km/h
    """

    fig_2d.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig_2d


def plot_trajectory_3d_interactive_multi(trajectories, initial_velocity, roughness, seam_angle):
    """Create interactive 3D plot with multiple trajectories."""
    if not PLOTLY_AVAILABLE:
        return None

    traces = []
    colors = ['blue', 'red', 'green', 'orange']

    for idx, traj in enumerate(trajectories):
        x, y, z = traj['x'], traj['y'], traj['z']
        name = traj['name']
        color = colors[idx % len(colors)]

        # Trajectory line
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=color, width=4),
            name=name,
            hovertemplate=f'{name}<br>Distance: %{{x:.2f}}m<br>Lateral: %{{y:.2f}}m<br>Height: %{{z:.2f}}m<extra></extra>'
        ))

        # Landing point
        traces.append(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode='markers',
            marker=dict(size=6, color=color, symbol='x'),
            showlegend=False,
            hovertext=f'{name} landing: {abs(y[-1]-y[0])*100:.1f} cm swing'
        ))

    # Release point (same for all)
    x0, y0, z0 = trajectories[0]['x'][0], trajectories[0]['y'][0], trajectories[0]['z'][0]
    traces.append(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=8, color='green', symbol='circle'),
        name='Release',
        hovertext=f'Release: {initial_velocity*3.6:.1f} km/h'
    ))

    # Pitch outline
    pitch_x = [0, 20.12, 20.12, 0, 0]
    pitch_y_left = [-1.22, -1.22, -1.22, -1.22, -1.22]
    pitch_y_right = [1.22, 1.22, 1.22, 1.22, 1.22]
    pitch_z = [0, 0, 0, 0, 0]

    traces.append(go.Scatter3d(
        x=pitch_x, y=pitch_y_left, z=pitch_z,
        mode='lines', line=dict(color='brown', width=2),
        showlegend=False, hoverinfo='skip'
    ))

    traces.append(go.Scatter3d(
        x=pitch_x, y=pitch_y_right, z=pitch_z,
        mode='lines', line=dict(color='brown', width=2),
        showlegend=False, hoverinfo='skip'
    ))

    traces.append(go.Scatter3d(
        x=[0, 20.12], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='black', width=1, dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f'Simplephysics vs Jaxphysics<br><sub>Roughness: {roughness:.2f}, Seam: {seam_angle:.0f}°, Speed: {initial_velocity*3.6:.0f} km/h</sub>',
        scene=dict(
            xaxis=dict(title='Distance (m)', range=[0, 22]),
            yaxis=dict(title='Lateral (m)', range=[-2, 2]),
            zaxis=dict(title='Height (m)', range=[0, 3]),
            aspectmode='manual',
            aspectratio=dict(x=2, y=0.5, z=0.3),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        width=1000,
        height=700,
        hovermode='closest'
    )

    return fig


def plot_optimization_results(speeds, devs, angles, swing_type, roughness):
    """Plot optimal swing and seam angle vs speed"""
    valid = ~np.isnan(devs)
    if not np.any(valid):
        print("No valid results to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Deviation
    color_dev = 'tab:blue'
    ax1.set_xlabel('Speed (m/s)')
    ax1.set_ylabel('Max Deviation (cm)', color=color_dev)
    ax1.plot(speeds[valid], devs[valid], color=color_dev, marker='o', label='Deviation')
    ax1.tick_params(axis='y', labelcolor=color_dev)
    ax1.grid(True, alpha=0.3)

    # Create secondary axis for Seam Angle
    ax2 = ax1.twinx()
    color_angle = 'tab:red'
    ax2.set_ylabel('Optimal Seam Angle (deg)', color=color_angle)
    ax2.plot(speeds[valid], angles[valid], color=color_angle, marker='s', linestyle='--', label='Seam Angle')
    ax2.tick_params(axis='y', labelcolor=color_angle)

    plt.title(f'Optimal {swing_type.capitalize()} Swing & Seam Angle (r={roughness})')
    fig.tight_layout()
    plt.show()
