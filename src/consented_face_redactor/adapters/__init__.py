"""adapters — face detection and embedding interface contracts.

The pipeline depends only on internal interfaces (DetectorAdapter,
EmbedderAdapter), never raw vendor output.  Implementations such as
YuNet / SFace or ONNXRuntime can be swapped without touching core logic.
"""
