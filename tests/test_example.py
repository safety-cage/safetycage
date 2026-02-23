# pytest convention that all files in ./tests begin with "test_"

def main():
    print("Hello from safetycage-tutorials!")


if __name__ == "__main__":
    main()


def test_add_one_basic():
    assert 1+1 == 2


# THIS IS NOT CONFIGURED PROPERLY RIGHT NOW


### create simple tests here, install the python package in the venv to quickly test,
# then use safetycage_tutorials to see a more end/user perspective
