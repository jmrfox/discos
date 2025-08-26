---
trigger: always_on
---

uv:
I use the uv package manager. When running Python in the terminal, do "uv run script.py" to run a script with the venv.
Tests should be run using "uv run pytest".

project structure:
all data goes in the data directory. 
all Jupyter notebooks go in the notebooks directory. 
all tests live in the tests directory. use "uv run pytest" to run tests.

code preferences:
never import a Python package that is not used in the code you write.

tests:
Keep test scripts (pytest) in the tests directory.
For classes and functions in the project codebase, we want to write tests that check that each method works as intended.
In the case that a test is failing, a common mistake is to alter the test so that it passes rather than change the code so it does what we intend. It is absolutely critical that tests check that code does what we intend. A test that passes without affirming code functionality must be avoided at all costs.