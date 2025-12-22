"""
Docker environment setup and management for tesseract containers
"""
import subprocess
import sys
import time
from tesseract_core import Tesseract


def check_docker_network(network_name: str) -> bool:
    """Check if a Docker network exists."""
    try:
        result = subprocess.run(
            ["docker", "network", "ls", "--filter",
                f"name={network_name}", "--format", "{{.Name}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        networks = result.stdout.strip().split("\\n")
        return network_name in networks
    except Exception as e:
        print(f"Failed to check Docker networks: {e}")
        return False


def ensure_docker_network(network_name: str) -> bool:
    """Create a Docker network if it does not exist."""
    print(f"\n🔍 Checking Docker network '{network_name}'...")
    try:
        inspect_result = subprocess.run(
            ["docker", "network", "inspect", network_name],
            capture_output=True,
            text=True
        )

        if inspect_result.returncode != 0:
            print(f"   Creating network '{network_name}'...")
            subprocess.run(
                ["docker", "network", "create", network_name],
                check=True,
                capture_output=True
            )
            print(f"✓ Network '{network_name}' created")
        else:
            print(f"✓ Network '{network_name}' exists")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def ensure_container(container_name: str, image_name: str,
                     network_name: str, host_port: int):
    """Start a container if not already running"""
    print(f"\n🔍 Checking {container_name} container...")

    # Check if already running
    check = subprocess.run(
        ["docker", "ps", "--filter", f"name={container_name}",
            "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    if container_name in check.stdout:
        print(f"✓ {container_name} is already running")
        return

    print(f"  Cleaning up old {container_name} container if any...")
    subprocess.run(["docker", "stop", container_name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", container_name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"  Starting {container_name} container on port {host_port}...")
    result = subprocess.run([
        "docker", "run", "-d",
        "--name", container_name,
        "--network", network_name,
        "--network-alias", container_name,
        "-p", f"{host_port}:8000",  # Map host_port to container's 8000
        image_name,
        "serve", "--host", "0.0.0.0", "--port", "8000"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed to start {container_name} container:")
        print(result.stderr)
        raise RuntimeError(f"Failed to start {container_name} container")

    print(f"✓ {container_name} container started")

    # Wait for it to be ready
    print(f"  Waiting for {container_name} to be ready...")
    max_wait = 30
    for i in range(max_wait):
        time.sleep(1)
        check = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}",
                "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if container_name not in check.stdout:
            print(f"❌ {container_name} container stopped unexpectedly")
            logs = subprocess.run(
                ["docker", "logs", container_name], capture_output=True, text=True)
            print(logs.stdout)
            print(logs.stderr)
            raise RuntimeError(
                f"{container_name} container stopped unexpectedly")

        health_check = subprocess.run(
            ["curl", "-f", f"http://localhost:{host_port}/health"],
            capture_output=True,
            text=True,
        )
        if health_check.returncode == 0:
            print(f"✓ {container_name} is ready (took {i+1}s)")
            break
        if i == max_wait - 1:
            print(f"⚠️  Warning: {container_name} may not be fully ready")
            logs = subprocess.run(
                ["docker", "logs", container_name], capture_output=True, text=True)
            print(logs.stdout[-500:] if logs.stdout else "No logs")


def ensure_simplephysics_container(network_name: str):
    """Start simplephysics container if not already running"""
    ensure_container(
        "simplephysics", "simplephysics", network_name, 8000)


def ensure_jaxphysics_container(network_name: str):
    """Start jaxphysics container if not already running"""
    ensure_container("jaxphysics", "jaxphysics", network_name, 8001)




def prepare_docker_environment(network_name: str, use_jaxphysics: bool = True):
    """Set up Docker environment including network and containers"""
    print("============================================================")
    print("  CRICKET BALL TRAJECTORY SIMULATION DEMO")
    print("============================================================")

    ensure_docker_network(network_name)
    ensure_simplephysics_container(network_name)

    if use_jaxphysics:
        ensure_jaxphysics_container(network_name)


def cleanup_containers():
    """Stop and remove containers"""
    global _INTEGRATOR, _SWING, _OPTIMIZER
    print("\n🧹 Cleaning up all containers...")
    
    # Reset globals
    _INTEGRATOR = None
    _SWING = None
    _OPTIMIZER = None

    containers = ["simplephysics", "jaxphysics", "integrator", "swing", "optimiser"]
    for container in containers:
        subprocess.run(["docker", "stop", container],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", container],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✓ Cleanup complete")


# Global tesseracts to avoid restarting containers on every call
_INTEGRATOR = None
_SWING = None
_OPTIMIZER = None


def get_tesseracts(network_name="tesseract_network"):
    """Get or setup tesseracts (singleton pattern)"""
    global _INTEGRATOR, _SWING, _OPTIMIZER
    if _INTEGRATOR is None:
        print("🔍 Starting Tesseracts (first time setup)...")
        _INTEGRATOR, _SWING, _OPTIMIZER = setup_tesseracts(network_name)
    return _INTEGRATOR, _SWING, _OPTIMIZER


def setup_tesseracts(network_name="tesseract_network"):
    """Setup all tesseracts (integrator, swing, optimizer)"""
    print("🚀 Setting up tesseracts...")

    # Start physics backends
    prepare_docker_environment(network_name, use_jaxphysics=False)

    # Start tesseracts with external port mappings
    ensure_container("integrator", "integrator", network_name, 8002)
    ensure_container("swing", "swing", network_name, 8003)
    ensure_container("optimiser", "optimiser", network_name, 8004)

    # Connect to running tesseracts
    integrator = Tesseract.from_url("http://localhost:8002")
    swing = Tesseract.from_url("http://localhost:8003")
    optimizer = Tesseract.from_url("http://localhost:8004")

    print("✓ Tesseracts ready")
    return integrator, swing, optimizer
