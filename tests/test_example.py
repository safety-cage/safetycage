import safetycage_testing
import numpy as np

# pytest convention that all files in ./tests begin with "test_"

def test_add_one_basic():
    assert 1+1 == 2

def test_check_package_version():
    from importlib.metadata import version
    assert version('safetycage_testing') == '0.1.6'


def main():
    test_add_one_basic()
    test_check_package_version()
    print("Hello from safetycage-tutorials!")

    print(np.NaN >= 0.05)

if __name__ == "__main__":
    main()


# THIS IS NOT CONFIGURED PROPERLY RIGHT NOW


### create simple tests here, install the python package in the venv to quickly test,
# then use safetycage_tutorials to see a more end/user perspective
