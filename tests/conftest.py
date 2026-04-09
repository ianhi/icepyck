from pathlib import Path

import pytest

TEST_REPOS_DIR = Path(__file__).parent.parent / "test-repos"


@pytest.fixture
def test_repos_dir():
    return TEST_REPOS_DIR


@pytest.fixture
def basic_repo(test_repos_dir):
    return test_repos_dir / "basic"


@pytest.fixture
def nested_repo(test_repos_dir):
    return test_repos_dir / "nested"


@pytest.fixture
def scalar_repo(test_repos_dir):
    return test_repos_dir / "scalar"


@pytest.fixture
def native_chunks_repo(test_repos_dir):
    return test_repos_dir / "native-chunks"
