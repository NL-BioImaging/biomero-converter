# Working conventions for this repo

Distilled from recurring feedback across sessions. Follow these when making changes here.

## Branches
- `main` is the release branch; `dev-new-rocrate` holds the RO-Crate work on top of it.
- Fixes to shared code go into both branches. Make the change on the checked-out branch, then
  cherry-pick onto the other in a temporary `git worktree`, so the user's checkout and any
  uncommitted changes stay untouched.
- RO-Crate tests (`test/test_rocrate.py`) stay off `main`.

## Releases
- Release only from `main`, with the next `v0.1.x` tag (`gh release create --generate-notes`, plus a
  short summary). Publishing the release builds the Docker image and the docs (`.github/workflows/docker-image.yml`).
- Only release when converter code changed. Test-only (or docs-only) changes need no release.
- The version is taken from the release tag, not from the code: `src/version.py` uses the
  `BIOMERO_CONVERTER_VERSION` env var (set in the Docker build from the tag) or `git describe --tags`.
  Don't add a version number anywhere else.
- After a release that fixes a GitHub issue, comment on the issue with the release link.

## Commits
- Commit and push only when asked, and only after the change is validated (tests / round-trip check).
- Small, focused commits with a "why" in the message; link issues (`Fixes #N`).

## Testing
- Test data lives in `C:/Project/slides` (Leica examples in `C:/Project/slides/Leica`).
- `test/test_convert.py`: each `filenames = [...]` line overrides the previous one; the last one is the active
  selection. Add new formats there rather than creating a new test file.
- Tests are about metadata variety: prefer small, varied files, run those first, and skip huge binary image
  examples (MIRAX `.mrxs`, iSyntax take 5-40+ min per format) unless asked.
- Run long test runs in the background, ordered small to large, and report as results come in.
- `test_convert` only compares pixel size (and wells), and only the first output of multi-image files: when
  changing a writer or source, also check the pixel data round-trips (read the output back with `create_source`).

## Sources
- New sources derive from `ImageSource`; follow `TiffSource` for structure (`init_metadata()` fills the
  attributes, getters return them). Register the extension in `src/helper.py`.
- Leica (`src/LeicaSource.py`, via `liffile`): follow https://github.com/NL-BioImaging/ConvertLeica-Docker for
  image selection (`image_uuid`) and tile scan stitching.

## Carrying work across sessions
- Record the current task, its plan and how far it got under "In progress" in `notes/todo_known_issues.md`
  before editing code, and keep it updated. Clear it once the task is done.
- On "continue" with no other context, read that section first and resume from it.
