## Workspace Notes

- Put temporary debug output created during investigation in `output/debug/` rather than the repository root.
- This applies in particular to ad hoc plots, PNGs, and other throwaway inspection artifacts.
- Never invoke `python` directly.
- Always use `./.conda/bin/python <cmd>`.
- When a user provides a file path, use it exactly.
- Do not substitute repo-relative paths unless explicitly instructed.
