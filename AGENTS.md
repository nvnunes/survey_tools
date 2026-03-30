## Workspace Notes

- Put temporary debug output created during investigation in `output/debug/` rather than the repository root.
- This applies in particular to ad hoc plots, PNGs, and other throwaway inspection artifacts.
- Never invoke `python` directly.
- Always use `./.conda/bin/python <cmd>`.
- When a user provides a file path, use it exactly.
- Do not substitute repo-relative paths unless explicitly instructed.
- Treat `data/maps/inner` and `data/maps/gaia` as symlink entry points, not fixed storage locations.
- Before reasoning about paths, file presence, or disk usage, resolve symlinks and use the real location.
- Treat top-level files under `data/maps` such as `outer.fits` and `data-hpx*.fits` as canonical direct files unless a symlink is explicitly present.
