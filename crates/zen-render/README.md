# zen-render

`zen-render` provides the domain-independent render host used by `zen-proto`.
`RenderHost<C>` owns FrameGraph compilation and execution, transient resource
pool lifetime, GPU timing, and optional Snapshot capture. A concrete
`FrameComposer` owns domain renderers and records their passes through typed
FrameGraph handles.

The host deliberately uses static generic composition. It does not define a
dynamic render-module registry or string-keyed resource blackboard. A composer
can connect Mesh, Line, Particle, post-processing, and other domains explicitly
while retaining their concrete input and output types.

## Lifecycle

```text
validate present format and extent
prepare_frame
  import and bind the present texture
  record_frame_graph
  mark_present
  compile and execute
after_submit                         successful execution only
after_discard                        any post-prepare failure
```

`PreparedFrame` is a transaction ticket. Exactly one terminal hook receives it:
`after_submit` after successful execution, or `after_discard` when surface
registration, recording, root declaration, compilation, or execution fails.
