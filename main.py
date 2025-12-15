"""
Main Pipeline - Tesseract Hackathon Template

This script demonstrates usage of the simplephysics tesseract for inference.

It instantiates the simplephysics tesseract and runs inference for a set of test inputs,
printing the resulting force vectors.
"""

import jax.numpy as jnp
from jax import Array
from tesseract_core import Tesseract


def simplephysics_tesseract_demo():
    print("\n" + "=" * 60)
    print("  SIMPLEPHYSICS TESSERACT INFERENCE DEMO")
    print("=" * 60)

    # Instantiate the simplephysics tesseract
    simplephysics = Tesseract.from_image("simplephysics")

    with simplephysics:
        # Define test input dictionaries as per simplephysics tesseract schema
        test_inputs = [
            {"roughness": 0.3, "notch_angle": -20, "reynolds_number": 2.5e5},
            {"roughness": 0.3, "notch_angle": 20, "reynolds_number": 2.5e5},
            {"roughness": 0.3, "notch_angle": 0, "reynolds_number": 2.5e5},
            {"roughness": 0.8, "notch_angle": -20, "reynolds_number": 2.5e5},
            {"roughness": 0.8, "notch_angle": 20, "reynolds_number": 2.5e5},
        ]

        for inp in test_inputs:
            output = simplephysics.apply(inp)
            forces = output["force_vector"]
            print(
                f"Input: roughness={inp['roughness']}, notch_angle={inp['notch_angle']}°, reynolds_number={inp['reynolds_number']}"
            )
            print(
                f"Output force vector: [drag={forces[0]:.6f}, lift={forces[1]:.6f}, side={forces[2]:.6f}]\n"
            )


def main() -> None:
    simplephysics_tesseract_demo()


if __name__ == "__main__":
    main()
