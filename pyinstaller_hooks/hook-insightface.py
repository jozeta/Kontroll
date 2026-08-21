"""PyInstaller data-file hook for insightface: bundles meanshape_68.pkl, the one file
insightface's own frozen-app code path (insightface/data/pickle_object.py::get_object) looks for
manually at runtime rather than importing.

Without this hook, a frozen build crashes on the first face capture with
`AttributeError: 'NoneType' object has no attribute 'shape'` deep inside
insightface/utils/transform.py::estimate_affine_matrix_3d23d -- not at startup, since model
*loading* (FaceAnalysis(...)) never touches this file; only landmark.py::Landmark.get() does,
which only runs once a face is actually detected in a captured frame. get_object() has an
explicit `if getattr(sys, 'frozen', False): base_dir = sys._MEIPASS` branch, meaning insightface
itself expects a frozen build to place this file at `<bundle root>/objects/meanshape_68.pkl` --
NOT the source layout's `insightface/data/objects/meanshape_68.pkl` -- which is exactly what the
`datas` mapping below does. See README.md "Building a standalone EXE" for the required
`--additional-hooks-dir` build flag, and CLAUDE.md's "Windows-specific choices" for more context.

PyInstaller auto-discovers this file by name (`hook-<package>.py`) when its directory is passed
via `--additional-hooks-dir`; nothing here needs to be imported or invoked directly.
"""

import os

from PyInstaller.utils.hooks import get_package_paths

_, _insightface_dir = get_package_paths("insightface")
_meanshape_pkl = os.path.join(_insightface_dir, "data", "objects", "meanshape_68.pkl")

# (source file, destination folder relative to the bundle root) -- destination is "objects", not
# "insightface/data/objects", to match get_object()'s frozen-mode lookup path exactly.
datas = [(_meanshape_pkl, "objects")]
