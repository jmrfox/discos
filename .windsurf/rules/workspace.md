---
trigger: always_on
---

all data goes in the data directory. 
all Jupyter notebooks go in the notebooks directory. 
all tests live in the tests directory. use "uv run pytest" to run tests.

never import a Python package that is not used in the code you write.