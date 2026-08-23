# 17 — README correction

**What to build:** A README that describes the library that exists. The current one describes a
library that does not.

Three concrete errors are already present, and they are the kind that waste a newcomer's first
half hour: the quick-start example constructs a second-order cone with a dimension argument the
class does not accept, the quality-checks section names a type checker that is not in the
toolchain, and the testing section points at a test directory that has never existed under that
name.

This is deliberately sequenced after the tracer bullet rather than before it, so the quick-start
example can show a real trajectory optimization problem being solved rather than a cone
projection standing in for one.

**Blocked by:** 09 — NLP transcription and the first Ipopt solve.

**Spec:** Section 3 (package structure) for the module layout the README describes; the
specification as a whole for the feature claims it makes.

## Acceptance criteria

- [x] Every code example in the README executes as written against the current API
- [x] The quick-start example solves a small trajectory optimization problem end to end rather
      than demonstrating an isolated primitive
- [x] The named type checker matches the one actually configured in the project
- [x] Test invocation paths match the real test layout
- [x] Dependency groups described in the README match those declared in the project metadata
- [x] The architecture section links to the specification at its current location
- [x] A test or lint step executes the README examples, so this class of drift is caught rather
      than rediscovered
