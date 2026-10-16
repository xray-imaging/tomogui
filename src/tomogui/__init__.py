__version__ = "1.0.0"

# Lazy re-export of TomoGUI so ``import tomogui.cli`` / ``import
# tomogui.headless`` don't drag in PyQt5, VisPy and the whole GUI stack.
# ``from tomogui import TomoGUI`` still works.
def __getattr__(name):
    if name == "TomoGUI":
        from .gui import TomoGUI
        return TomoGUI
    raise AttributeError(f"module 'tomogui' has no attribute {name!r}")


__all__ = ["TomoGUI"]