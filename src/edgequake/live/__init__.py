"""EdgeQuake live engine: pluggable waveform sources + streaming EEW pipeline.

Design (Phase 4):
    Source (replay / SeedLink / FDSN-poll)  ->  Packets
    LiveEngine: ring buffers -> PhaseNet picks -> event association
                -> location + magnitude -> county intensities + PWS
    state.json (atomic write)  ->  polled by web/live.html console

The engine runs on a local machine (a laptop is enough — picking is ~5-10 ms
per station-window on CPU). The web console is a static page that polls
state.json, so the same frontend works from a local HTTP server today and a
Vercel deployment (with a small state store) later.
"""
