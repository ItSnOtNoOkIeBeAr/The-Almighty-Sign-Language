# DECISIONS

One line per non-obvious choice: what and why.

## Phase 0
- **Repo location `C:\dev\almighty-sign-language`**: the user's home path contains a space, which breaks some CUDA/CMake tooling (spec rule 8).
- **VS dev environment loaded by script, not by reopening VS Code**: `cl` and `cmake` are not on the default PATH. Scripts call `vcvars64.bat` (located via the VS install path) so builds work from any terminal. Verified: `cl` 19.42.34436, CMake 3.29.5-msvc4, Ninja from VS.
- **Extra error codes in PROTOCOL.md** (`bad_type`, `too_short`, `bad_version`, `empty_payload`, `internal`): the spec lists `bad_magic|too_large|decode_failed|...`; these fill in the `...` so both servers and the conformance test agree on exact codes.
- **End-of-stream semantics**: spec defines the flag but not the behaviour. Chosen: an EOS message with an empty payload gets no result and resets that connection's decision state.
- **Max message size 2 MB = 2 097 152 bytes, header included**.
