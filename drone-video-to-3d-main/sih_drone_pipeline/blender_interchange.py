"""Executed only by Blender's bundled Python."""
import sys
from pathlib import Path

if __name__ == "__main__":
    import bpy
    source, destination = sys.argv[sys.argv.index("--") + 1:]
    destination = Path(destination)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=source)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.ops.export_scene.gltf(filepath=str(destination / "scene.gltf"), export_format="GLTF_SEPARATE")
    bpy.ops.export_scene.fbx(filepath=str(destination / "scene.fbx"), path_mode="COPY", embed_textures=True,
                             bake_anim=False, add_leaf_bones=False)
