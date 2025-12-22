"""
Minimal Main - Tesseract Cricket Ball Simulator

Functions:
- show_trajectory: Display trajectory for given parameters
- plot_optimal: Plot optimal swing vs speed and notch_angle
"""
import numpy as np
import matplotlib.pyplot as plt
from helper.docker import setup_tesseracts, cleanup_containers
from plotter.plt import plot_trajectory_3d


def show_trajectory(initial_velocity=35.0, release_angle=5.0, roughness=0.8, seam_angle=30.0,
                   physics_url="http://simplephysics:8000"):
    """
    Show trajectory for given parameter set

    Args:
        initial_velocity: Ball speed in m/s
        release_angle: Release angle in degrees
        roughness: Surface roughness [0.0, 1.0]
        seam_angle: Seam angle in degrees [-90, 90]
        physics_url: Physics backend URL
    """
    network_name = "tesseract_network"

    try:
        print(f"🚀 Setting up Docker environment...")
        integrator, swing, optimizer = setup_tesseracts(network_name)

        print(f"📡 Connecting to integrator...")
        with integrator:
            print("\n📊 Computing trajectory...")
            print(f"  Speed: {initial_velocity:.1f} m/s")
            print(f"  Release angle: {release_angle:.1f}°")
            print(f"  Roughness: {roughness:.1f}")
            print(f"  Seam angle: {seam_angle:.1f}°")

            # Get trajectory
            params = {
                "initial_velocity": initial_velocity,
                "release_angle": release_angle,
                "roughness": roughness,
                "seam_angle": seam_angle,
                "physics_url": physics_url
            }

            print(f"🔄 Calling integrator with params: {params}")
            result = integrator.apply(params)
            print(f"✅ Integrator returned result with {len(result.get('times', []))} time points")

            # Plot trajectory
            fig = plot_trajectory_3d(
                result["times"],
                result["x_positions"],
                result["y_positions"],
                result["z_positions"],
                result.get("velocities"),
                initial_velocity,
                roughness,
                seam_angle,
                use_plotly=True
            )
            fig.show()

            # Print summary
            final_y = result["y_positions"][-1]
            print(f"  Final lateral deviation: {final_y:.4f} m")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cleanup_containers()


def plot_optimal(roughness=0.8, release_angle=5.0, swing_type="reverse",
                speed_range=[30, 40], angle_range=[-45, 45],
                n_points=10):
    """
    Plot optimal swing vs speed and notch_angle for specified roughness and release angle

    Args:
        roughness: Fixed surface roughness
        release_angle: Fixed release angle in degrees
        swing_type: "in", "out", or "reverse"
        speed_range: [min_speed, max_speed] in m/s
        angle_range: [min_angle, max_angle] in degrees
        n_points: Number of points to sample in each dimension
    """
    network_name = "tesseract_network"

    try:
        _, swing, optimizer = setup_tesseracts(network_name)

        with optimizer:
            print(f"\n🎯 Optimizing {swing_type} swing...")
            print(f"  Roughness: {roughness:.1f}")
            print(f"  Release angle: {release_angle:.1f}°")
            # Create parameter grid
            speeds = np.linspace(speed_range[0], speed_range[1], n_points)
            angles = np.linspace(angle_range[0], angle_range[1], n_points)

            # Store results
            optimal_deviations = np.zeros((n_points, n_points))

            for i, speed in enumerate(speeds):
                for j, angle in enumerate(angles):
                    try:
                        # Optimize for this fixed speed/angle combination
                        opt_result = optimizer.apply({
                            "fixed_variables": {
                                "initial_velocity": speed,
                                "release_angle": release_angle,
                                "roughness": roughness
                            },
                            "optimization_variables": {
                                "seam_angle": [angle_range[0], angle_range[1]]  # Optimize seam angle
                            },
                            "swing_type": swing_type,
                            "swing_url": "http://swing:8000",
                            "physics_url": "http://simplephysics:8000",
                            "integrator_url": "http://integrator:8000"
                        })

                        optimal_deviations[i, j] = opt_result["maximum_deviation"]

                    except Exception as e:
                        print(f"⚠️  Optimization failed for speed={speed:.1f}, angle={angle:.1f}: {e}")
                        optimal_deviations[i, j] = np.nan

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create meshgrid
            SPEED, ANGLE = np.meshgrid(speeds, angles)

            # Plot contour
            cs = ax.contourf(SPEED, ANGLE, optimal_deviations.T, levels=20, cmap='viridis')
            ax.set_xlabel('Initial Velocity (m/s)')
            ax.set_ylabel('Fixed Seam Angle (degrees)')
            ax.set_title(f'Optimal {swing_type.capitalize()} Swing (Roughness: {roughness:.1f})')
            plt.colorbar(cs, ax=ax, label='Maximum Deviation (cm)')

            # Mark optimal point
            max_idx = np.unravel_index(np.nanargmax(optimal_deviations), optimal_deviations.shape)
            max_speed, max_angle = speeds[max_idx[0]], angles[max_idx[1]]
            max_dev = optimal_deviations[max_idx]

            ax.plot(max_speed, max_angle, 'ro', markersize=10, label=f'Optimal: {max_dev:.2f} cm')
            ax.legend()

            plt.tight_layout()
            plt.show()

            print(f"\n💡 Global optimum found:")
            print(f"  Speed: {max_speed:.1f} m/s")
            print(f"  Seam angle: {max_angle:.1f}°")
            print(f"  Maximum deviation: {max_dev:.2f} cm")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_containers()


def interactive_menu():
    """Interactive menu to choose between functions"""
    print("🎾 Cricket Ball Swing Simulator")
    print("=" * 40)
    print("Choose an option:")
    print("1. Show trajectory for specific parameters")
    print("2. Plot optimal swing landscape")
    print("3. Exit")

    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()

            if choice == "1":
                print("\n📊 Show Trajectory")
                print("-" * 20)

                # Get parameters with defaults
                vel = input("Initial velocity (m/s) [35.0]: ").strip()
                vel = float(vel) if vel else 35.0

                angle = input("Release angle (degrees) [5.0]: ").strip()
                angle = float(angle) if angle else 5.0

                rough = input("Roughness [0.8]: ").strip()
                rough = float(rough) if rough else 0.8

                seam = input("Seam angle (degrees) [30.0]: ").strip()
                seam = float(seam) if seam else 30.0

                show_trajectory(vel, angle, rough, seam)

            elif choice == "2":
                print("\n🎯 Plot Optimal Swing")
                print("-" * 25)

                # Get parameters with defaults
                rough = input("Roughness [0.8]: ").strip()
                rough = float(rough) if rough else 0.8

                angle = input("Release angle (degrees) [5.0]: ").strip()
                angle = float(angle) if angle else 5.0

                swing_type = input("Swing type (in/out/reverse) [reverse]: ").strip().lower()
                swing_type = swing_type if swing_type in ["in", "out", "reverse"] else "reverse"

                print(f"Generating plot for {swing_type} swing...")
                plot_optimal(rough, angle, swing_type)

            elif choice == "3":
                print("\n👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")


def main():
    """Main entry point - run interactive menu"""
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
