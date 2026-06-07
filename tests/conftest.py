import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _pytest._py.path import LocalPath

# derived from https://github.com/elifesciences/sciencebeam-trainer-delft/tree/develop/tests

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    logging.root.handlers = []
    logging.basicConfig(level="INFO")
    logging.getLogger("tests").setLevel("DEBUG")
    # logging.getLogger('sciencebeam_trainer_delft').setLevel('DEBUG')


def _backport_assert_called(mock: MagicMock):
    assert mock.called


@pytest.fixture(scope="session", autouse=True)
def patch_magicmock():
    try:
        MagicMock.assert_called
    except AttributeError:
        MagicMock.assert_called = _backport_assert_called


@pytest.fixture
def temp_dir(tmpdir: LocalPath):
    # convert to standard Path
    return Path(str(tmpdir))
