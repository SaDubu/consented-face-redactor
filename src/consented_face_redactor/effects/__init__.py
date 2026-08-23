"""Effects module — mosaic, sticker renderers.

No raw image paths, human names, crops, or debug data leak into tests.
Renderers NEVER mutate their input arrays.
"""

from .mosaic import MosaicEffect
from .sticker import StickerEffect

__all__ = ["MosaicEffect", "StickerEffect"]
