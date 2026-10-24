import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    print()