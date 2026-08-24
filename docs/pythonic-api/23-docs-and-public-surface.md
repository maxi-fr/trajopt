# 23 — Docs describe the finished API

**What to build:** nothing in the repo documents a name that no longer exists. A reader coming
to the spec or the README finds the API they will actually import, and the package exposes one
public surface instead of two.

**Blocked by:** 22.

## Why

The specification pins the old free-function API explicitly — the MPC methods section lists
`cost(problem, state)`, `rollout(problem, state)`, `states(state)`, `controls(state)`,
`initial_states(state, X0)`, and the control-loop pseudocode is written in that idiom. The
README opens its usage example with the exact snippet this refactor set out to eliminate. A
specification describing a dead API is worse than no specification, because it is believed.

This lands last so the docs are rewritten once against the finished state rather than six times
against moving targets.

## What changes

The specification's MPC section and control-loop pseudocode are rewritten to the shipped API.
Its design-decisions table already claims "Pythonic naming, no Julia bang-suffix mirroring" —
make that true rather than aspirational. Where the spec describes trajectory accessors that
ticket 21 deleted, describe the fields instead.

The README usage example uses the method form.

The top-level package `__init__.py` is emptied but for the environment import. The project
convention says `__init__.py` stays empty and callers import from the module; the top-level one
is currently around 230 lines re-exporting roughly 130 names, which is a second API surface to
keep in sync with every rename in this sequence. The blast radius is small — the tests already
import from submodules almost everywhere, with only two files importing the package bare.

Keep the environment import, with a comment saying why: it registers the Ipopt DLL directory on
Windows at import time, so removing it breaks Ipopt for Windows users in a way that will not be
obvious to whoever "cleans it up" next.

## Acceptance criteria

- [ ] The specification's MPC methods section and control-loop pseudocode describe the shipped API.
- [ ] The specification mentions no deleted name.
- [ ] The README usage example uses the method form.
- [ ] The top-level `__init__.py` contains only the environment import, with a comment explaining why it must stay.
- [ ] The two test files importing the package bare are updated to import from submodules.
- [ ] Ipopt still resolves on Windows from a fresh interpreter.
- [ ] Full suite green with no assertion changed.
