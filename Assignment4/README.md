# Assignment 4 | EECS 648

Deliverable for screenshot is in [`./CompareFeatureSelectionModelsOutput.pdf`](./CompareFeatureSelectionModelsOutput.pdf)
Deliverable for Q4 is at [`./Delieverable4POV.xlsx`](./Delieverable4POV.xlsx)
Delieverable for Q5 is at [`./REPORT.md`](./REPORT.md)

--- 

[ `poetry` ](https://python-poetry.org/) is used for dependency management

## Pre-reqs

* `python 3.12^.0`
  * Probably doesn't work on lower verisons, some type features are relatively new... Would require `__future__` for backporting

## Run

```sh
python main.py
```

or with Poetry (in pwsh)
```pwsh
poetry run python .\main.py
```

<!-- ## Tests

Test cases are written using [ `unittest` ](https://docs.python.org/3/library/unittest.html) for testing. The tests can be found in [ `./tests` ](tests).

Run tests with python tests/main.py -v -->

## Formatting
[`black`](https://github.com/psf/black) is used for formatting and code-styling.

```sh
black .
```