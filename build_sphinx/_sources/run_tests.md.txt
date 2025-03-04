# Run Tests

To run the desired tests, you need to:

- Select the desired test case
- Select the tests to run on the chosen test case
- Run the test cases

## Select Test Case

You need to run the file `tests_FEniCS/select_testcase.py` which will generate a json configuration file.

The available test case are :

* **1D testcases :**
    * 1D Poisson problem with Dirichlet BC
    * 1D general elliptic system and convection-dominated regime with Dirichlet BC

* **2D testcases :**
    * 2D Poisson problem with low frequency in a square domain with Dirichlet BC
    * 2D Poisson problem with high frequency in a square domain with Dirichlet BC
    * 2D anisotropic elliptic problem on a square domain with Dirichlet BC
    * 2D Poisson problem on an annulus with mixed boundary conditions

For certain test cases, a choice of different versions will be proposed, corresponding to the options chosen for the prediction.

## Select Tests

Once you have selected the test case on which you want to run a certain number of tests, you need to run the `tests_FEniCS/select_tests.py` script, which will also generate a json configuration file. Depending on the test case selected, this will ask a number of questions, enabling you to select which tests to run (error estimates, gains, etc.).

## Run Tests

Then simply run the `tests_FEniCS/run_tests.py` file, which, starting from a `tests.json` configuration file, will execute all the tests required (for a specific test case). 

By default, results that have already been run will simply be read. To re-run results even if they already exist, please add the `--new_run` option.

To avoid plots being displayed when the script is run :

```sh
MPLBACKEND=Agg python3 run_tests.py
```