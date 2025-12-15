# ============================================================================
# INTERACTIVE DEMO
# ============================================================================

from tesseract_core import Tesseract
from .plt import plot_trajectory_3d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Predefined scenarios
scenarios = {
    '1': {
        'name': 'Outswinger (Fast bowler)',
        'velocity': 44.4,
        'angle': 0,
        'roughness': 0.3,
        # Negative angle = outswing (away from batsman, +y)
        'seam_angle': -20,
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


def run_interactive_demo():
    """Run interactive cricket ball trajectory simulator."""
    print("=" * 70)
    print("CRICKET BALL TRAJECTORY SIMULATOR")
    print("Using Tesseract for Aerodynamics")
    print("=" * 70)

    # Get the integrator tesseract
    integrator = Tesseract.from_image("integrator")

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
        seam_angle = float(
            input("  Seam angle (-90 to +90, negative=outswing, positive=inswing): "))
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
    with integrator:
        # Get trajectory
        result = integrator.apply({
            "initial_velocity": velocity,
            "release_angle": angle,
            "roughness": roughness,
            "seam_angle": seam_angle
        })

    print(f"✓ Simulation complete! ({len(result.x_positions)} points)")

    # Debug: print actual y values
    print(f"\nTrajectory check:")
    print(
        f"  Y-position range: {result.y_positions.min():.4f} to {result.y_positions.max():.4f} meters")
    print(
        f"  Total lateral swing: {abs(result.y_positions[-1] - result.y_positions[0])*100:.2f} cm")

    # Create visualizations
    print("\nGenerating visualizations...")
    fig_2d = plot_trajectory_3d(
        result.times,
        result.x_positions,
        result.y_positions,
        result.z_positions,
        result.velocities,
        velocity,
        roughness,
        seam_angle,
        use_plotly=True
    )

    plt.show()

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


def compare_deliveries():
    """Compare multiple delivery types side by side."""
    # Get the integrator tesseract
    integrator = Tesseract.from_image("integrator")

    # Use first 4 scenarios (skip custom)
    deliveries = [(s['name'], s['velocity'], s['angle'], s['roughness'], s['seam_angle'])
                  for k, s in scenarios.items() if k != '5']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    with integrator:
        for idx, (name, vel, ang, rough, seam) in enumerate(deliveries):
            result = integrator.apply({
                "initial_velocity": vel,
                "release_angle": ang,
                "roughness": rough,
                "seam_angle": seam
            })

            ax = axes[idx]
            ax.plot(result.x_positions, result.y_positions,
                    linewidth=2, label='Trajectory')
            ax.scatter(result.x_positions[0], result.y_positions[0],
                       c='green', s=100, marker='o', label='Release')
            ax.scatter(result.x_positions[-1], result.y_positions[-1],
                       c='red', s=100, marker='x', label='Landing')

            # Pitch
            pitch_width = 2.44
            ax.add_patch(Rectangle((0, -pitch_width/2), 20.12, pitch_width,
                                   fill=False, edgecolor='brown', linewidth=2))
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Lateral deviation (m)')
            ax.set_title(
                f'{name}\n{vel*3.6:.0f} km/h, Rough={rough:.1f}, Seam={seam}°')
            ax.set_xlim(-1, 22)
            ax.set_ylim(-1.5, 1.5)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add swing measurement
            swing = abs(result.y_positions[-1] - result.y_positions[0]) * 100
            ax.text(0.95, 0.05, f'Swing: {swing:.1f} cm',
                    transform=ax.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.show()
