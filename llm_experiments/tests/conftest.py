import pytest
import sys
from pathlib import Path

# Automatically add the project's 'src' directory to Python's path for all tests
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from settings import settings
from infrastructure.sidra import SidraClient
from infrastructure.cache import DistributionCache

@pytest.fixture(scope="session")
def sidra_client():
    """
    Session-scoped fixture to provide a SidraClient.
    """
    with SidraClient() as client:
        yield client

@pytest.fixture(scope="session")
def db_cache():
    """
    Session-scoped fixture to provide the local SQLite DistributionCache.
    """
    db_path = project_root / settings.paths.distributions_db
    with DistributionCache(db_path) as cache:
        yield cache
