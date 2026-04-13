# AGENTS instruction

## **REQUIRED: AI/Agent Disclosure**

Every file which was created, or altered, should have an additional comment at the top
saying "AI modified: " followed by a timestamp and a hash of the parent commit. This way
it's clear for the reviewer which files were AI generated.

This is a **mandatory requirement**. Include it with every modification to any file. If
there are existing AI comments, add yours right after them.

## Coding style

- Don't explain every line with a separate comment, use comments for complex chunks of code,
  or DSL logic. Don't write docstrings for single-line or simple functions/methods.
- Use structures and code style that are already present in the codebase. Don't introduce
  another approach for new changes.
- Read `CONTRIBUTING.md` file for more details how to build the project and run tests.

## Data Model
- When working with DataLoader and other code that saves data to a file directory, read the DATA_MODEL.md file and
  ensure that the resulting directory structure is compatible with it.


## Testing
- After major code changes run the relevant portion of the test suite 'poetry run pytest ...', run mypy and fix error, and run pre-commit and fix issues.
