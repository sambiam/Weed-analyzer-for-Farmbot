# Draw shape (archived)

The "Draw shape" tab was removed from the app. It let a user hand-edit and
send raw firmware G-code straight to the Farmduino via FarmBot OS's Lua
`gcode()` escape hatch, bypassing all of FarmBot OS's motion planning and
safety checks (see [[raw-gcode-draw-shape]]).

This directory keeps `shape_gcode.py` (shape/G-code generation) and its
tests for reference in case the feature is reinstated. The web routes
(`/draw-shape`, `/api/draw-shape/plan`, `/api/draw-shape/run`,
`/api/draw-shape/status`), their JS, and the supporting models
(`DrawShapePlanRequest`, `DrawShapeRunRequest`, `GcodeRunRequest`,
`GcodeRunStatus`) and `HomeAssistantClient` methods (`start_gcode_run`,
`gcode_run_status`) were deleted from the live app rather than archived,
since they were thin plumbing around this module.

This folder is excluded from ruff/pytest collection — it is not part of the
shipped app.
