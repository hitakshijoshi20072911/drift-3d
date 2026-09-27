"""Geometry backbone adapter used by DRIFTX product code."""


def load_backbone(model_name: str = "depth-anything/DA3-LARGE-1.1"):
    """Load the frozen pretrained geometry backbone on demand."""
    try:
        from third_party.depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError as exc:
        if not str(exc).startswith("No module named 'third_party"):
            raise
        from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(model_name)
