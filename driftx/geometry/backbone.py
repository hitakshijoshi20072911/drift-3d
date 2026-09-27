"""Geometry backbone adapter used by DRIFTX product code."""


def load_backbone(model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE"):
    """Load the frozen pretrained geometry backbone on demand."""
    from third_party.depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(model_name)
