# pytest convention that all files in ./tests begin with "test_"

from safetycage_testing.example import add_one # currently doesn't work

def test_add_one_basic():
    assert add_one(1) == 2

def test_add_one_negative():
    assert add_one(-1) == 0


### create simple tests here, install the python package in the venv to quickly test,
# then use safetycage_tutorials to see a more end/user perspective
