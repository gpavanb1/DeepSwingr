import numpy as np
import matplotlib.pyplot as plt
from helper.docker import get_tesseracts, cleanup_containers, PHYSICS_BACKENDS, switch_backend
from plotter.plt import plot_trajectory_3d, plot_optimization_results

def show_trajectory(v=35.0, a=5.0, r=0.8, s=30.0, backend="simplephysics"):
    """Show trajectory for given parameters"""
    try:
        url = PHYSICS_BACKENDS[backend]["url"]
        integrator, _, _ = get_tesseracts(backend_name=backend)
        res = integrator.apply({"initial_velocity": v, "release_angle": a, "roughness": r, "seam_angle": s, "physics_url": url})
        plot_trajectory_3d(res["times"], res["x_positions"], res["y_positions"], res["z_positions"], 
                           res.get("velocities"), v, r, s, use_plotly=True).show()
    except Exception as e:
        print(f"Error: {e}")

def plot_optimal(r=0.8, a=5.0, swing_type="out", speed_range=[30, 40], n=5, backend="simplephysics"):
    """Plot optimal swing and seam angle vs speed"""
    try:
        url = PHYSICS_BACKENDS[backend]["url"]
        _, _, opt = get_tesseracts(backend_name=backend)
        print(f"Optimizing {swing_type} swing using {backend}...")
        speeds = np.linspace(speed_range[0], speed_range[1], n)
        devs = []
        angles = []
        for i, v in enumerate(speeds):
            print(f"  [{i+1}/{n}] {v:.1f} m/s:", end=" ", flush=True)
            try:
                res = opt.apply({
                    "fixed_variables": {"initial_velocity": v, "release_angle": a, "roughness": r},
                    "optimization_variables": {"seam_angle": [-90, 90]},
                    "swing_type": swing_type,
                    "physics_url": url
                })
                val = res["maximum_deviation"]
                angle = res["optimal_parameters"]["seam_angle"]
                devs.append(val)
                angles.append(angle)
                print(f"{val:.2f} cm (at {angle:.1f}°)")
            except Exception as e:
                print(f"Failed: {e}")
                devs.append(np.nan)
                angles.append(np.nan)
        
        devs = np.array(devs)
        angles = np.array(angles)
        plot_optimization_results(speeds, devs, angles, swing_type, r)
    except Exception as e:
        print(f"Error: {e}")

def main():
    try:
        # Ask for physics backend at the start
        print("\n⚙️ Configure Physics Backend")
        backends = list(PHYSICS_BACKENDS.keys())
        for i, b in enumerate(backends):
            print(f"{i+1}. {b}")
        
        choice_idx = int(input(f"Choice [1]: ").strip() or "1") - 1
        if 0 <= choice_idx < len(backends):
            backend = backends[choice_idx]
        else:
            backend = "simplephysics"
        
        print(f"Using {backend} physics backend.")

        while True:
            print(f"\n🎾 Cricket Ball Simulator (Backend: {backend})")
            print("1. Trajectory\n2. Optimal Swing\n3. Change Physics Backend\n4. Exit")
            choice = input("Choice: ").strip()
            if choice == "4":
                break
            
            if choice == "3":
                print("\n⚙️ Change Physics Backend")
                for i, b in enumerate(backends):
                    print(f"{i+1}. {b}")
                choice_idx = int(input(f"Choice [1]: ").strip() or "1") - 1
                if 0 <= choice_idx < len(backends):
                    backend = switch_backend(backends[choice_idx])
                continue

            try:
                if choice == "1":
                    show_trajectory(
                        float(input("Initial velocity (m/s) [35.0]: ") or 35.0),
                        float(input("Release angle (deg) [5.0]: ") or 5.0),
                        float(input("Surface roughness (0-1) [0.8]: ") or 0.8),
                        float(input("Seam angle (deg) [30.0]: ") or 30.0),
                        backend=backend
                    )
                elif choice == "2":
                    plot_optimal(
                        r=float(input("Surface roughness (0-1) [0.8]: ") or 0.8),
                        swing_type=input("Swing type (in/out) [out]: ").lower() or "out",
                        backend=backend
                    )
            except Exception as e:
                print(f"Error: {e}")
    finally:
        cleanup_containers()

if __name__ == "__main__":
    main()
