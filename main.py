import numpy as np
import matplotlib.pyplot as plt
from helper.docker import setup_tesseracts, cleanup_containers
from plotter.plt import plot_trajectory_3d

def show_trajectory(v=35.0, a=5.0, r=0.8, s=30.0, url="http://simplephysics:8000"):
    """Show trajectory for given parameters"""
    try:
        integrator, _, _ = setup_tesseracts("tesseract_network")
        with integrator:
            res = integrator.apply({"initial_velocity": v, "release_angle": a, "roughness": r, "seam_angle": s, "physics_url": url})
            plot_trajectory_3d(res["times"], res["x_positions"], res["y_positions"], res["z_positions"], 
                               res.get("velocities"), v, r, s, use_plotly=True).show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup_containers()

def plot_optimal(r=0.8, a=5.0, swing_type="out", speed_range=[30, 40], n=5):
    """Plot optimal swing vs speed"""
    try:
        _, _, opt = setup_tesseracts("tesseract_network")
        with opt:
            print(f"Optimizing {swing_type} swing...")
            speeds = np.linspace(speed_range[0], speed_range[1], n)
            devs = []
            for i, v in enumerate(speeds):
                print(f"  [{i+1}/{n}] {v:.1f} m/s:", end=" ", flush=True)
                try:
                    res = opt.apply({
                        "fixed_variables": {"initial_velocity": v, "release_angle": a, "roughness": r},
                        "optimization_variables": {"seam_angle": [-45, 45]},
                        "swing_type": swing_type
                    })
                    val = res["maximum_deviation"]
                    devs.append(val)
                    print(f"{val:.2f} cm")
                except Exception as e:
                    print(f"Failed: {e}")
                    devs.append(np.nan)
            
            devs = np.array(devs)
            valid = ~np.isnan(devs)
            if np.any(valid):
                plt.figure(figsize=(8, 5))
                plt.plot(speeds[valid], devs[valid], 'b-o')
                plt.xlabel("Speed (m/s)")
                plt.ylabel("Max Deviation (cm)")
                plt.title(f"Optimal {swing_type.capitalize()} Swing (r={r})")
                plt.grid(True, alpha=0.3)
                plt.show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup_containers()

def main():
    while True:
        print("\n🎾 Cricket Ball Simulator\n1. Trajectory\n2. Optimal Swing\n3. Exit")
        choice = input("Choice: ").strip()
        if choice == "3":
            break
        try:
            if choice == "1":
                show_trajectory(
                    float(input("Initial velocity (m/s) [35.0]: ") or 35.0),
                    float(input("Release angle (deg) [5.0]: ") or 5.0),
                    float(input("Surface roughness (0-1) [0.8]: ") or 0.8),
                    float(input("Seam angle (deg) [30.0]: ") or 30.0)
                )
            elif choice == "2":
                plot_optimal(
                    r=float(input("Surface roughness (0-1) [0.8]: ") or 0.8),
                    swing_type=input("Swing type (in/out) [out]: ").lower() or "out"
                )
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
