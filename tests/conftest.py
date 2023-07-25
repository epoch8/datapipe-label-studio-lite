import tempfile
from pathlib import Path
import pytest
import os

import pandas as pd
from sqlalchemy import create_engine

from datapipe.store.database import DBConn
from datapipe_label_studio_lite.sdk_utils import sign_up, login_and_get_token


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        yield d


def assert_idx_equal(a, b):
    a = sorted(list(a))
    b = sorted(list(b))

    assert a == b


def assert_df_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    assert_idx_equal(a.index, b.index)

    eq_rows = (a == b).all(axis="columns")

    if eq_rows.all():
        return True

    else:
        print("Difference")
        print("A:")
        print(a.loc[-eq_rows])
        print("B:")
        print(b.loc[-eq_rows])

        raise AssertionError


@pytest.fixture
def dbconn():
    DBCONNSTR = "sqlite+pysqlite3:///:memory:"
    DB_TEST_SCHEMA = None

    if DB_TEST_SCHEMA:
        eng = create_engine(DBCONNSTR)

        try:
            eng.execute(f"DROP SCHEMA {DB_TEST_SCHEMA} CASCADE")
        except Exception:
            pass

        eng.execute(f"CREATE SCHEMA {DB_TEST_SCHEMA}")

        dbconn = DBConn(DBCONNSTR, DB_TEST_SCHEMA)

    else:
        dbconn = DBConn(DBCONNSTR, DB_TEST_SCHEMA)

    yield dbconn

    if DB_TEST_SCHEMA:
        eng.execute(f"DROP SCHEMA {DB_TEST_SCHEMA} CASCADE")


ls_host = os.environ.get("LABEL_STUDIO_HOST", "localhost")
ls_port = os.environ.get("LABEL_STUDIO_PORT", "8080")
ls_url = f"http://{ls_host}:{ls_port}/"
api_key = sign_up(ls_url, "test@epoch8.com", "qwerty123")
if api_key is None:
    api_key = login_and_get_token(ls_url, "test@epoch8.com", "qwerty123")


@pytest.fixture
def ls_url_and_api_key(tmp_dir):
    yield ls_url, api_key
