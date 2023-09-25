from typing import List
from datapipe.datatable import DataStore
from datapipe.store.database import TableStoreDB


def check_columns_are_in_table(ds: DataStore, tbl_name: str, columns: List[str]):
    datatable = ds.get_table(tbl_name)
    assert isinstance(datatable.table_store, TableStoreDB)
    for column in columns:
        if column not in [x.name for x in datatable.table_store.data_sql_schema]:
            raise ValueError(f"Missing '{column}' column in table {tbl_name}")
    return True
