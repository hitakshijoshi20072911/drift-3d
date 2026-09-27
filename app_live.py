import sys
from pathlib import Path

import trimesh


def main():
    if len(sys.argv) != 2:
        print('Usage: python app_live.py "path\\to\\scene.glb"')
        return 1

    glb = Path(sys.argv[1]).expanduser().resolve()

    if not glb.exists():
        print(f"[ERROR] GLB not found: {glb}")
        return 1

    scene = trimesh.load(
        glb,
        force="scene",
        process=False,
    )

    print(f"[DRIFT] Loaded: {glb}")
    print(f"[DRIFT] Geometry objects: {len(scene.geometry)}")

    if scene.bounds is not None:
        minimum, maximum = scene.bounds
        dimensions = maximum - minimum
        diagonal = float((dimensions @ dimensions) ** 0.5)

        print(f"[DRIFT] Bounding-box dimensions: {dimensions}")
        print(f"[DRIFT] Bounding-box diagonal: {diagonal:.4f} scene units")

    # Correct trimesh 5.x viewer API
    from trimesh.viewer import windowed

    windowed.SceneViewer(
        scene=scene,
        smooth=False,
        resolution=(1440, 900),
        resizable=True,
        caption="DRIFT — DA3 3D Reconstruction",
        flags={
            "cull": False,
            "wireframe": False,
        },
        start_loop=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())