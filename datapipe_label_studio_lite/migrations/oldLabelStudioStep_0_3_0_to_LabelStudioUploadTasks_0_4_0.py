import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from datapipe.datatable import DataStore
from datapipe.store.database import DBConn, TableStoreDB
from label_studio_sdk import LabelStudio
from label_studio_sdk.data_manager import DATETIME_FORMAT
from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, Table, inspect, text

from datapipe_label_studio_lite.sdk_utils import (
    get_project_by_title,
    get_tasks_iter,
    login_and_get_token,
    project_to_dict,
)
from datapipe_label_studio_lite.types import ProjectDict

logger = logging.getLogger("datapipe_label_studio_lite.migrations")


@dataclass
class LabelStudioMigrationSpec:
    input_table: str
    output_table: str
    sync_table: str
    primary_keys: List[str]
    columns: List[str]

    project_identifier: Union[str, int]
    ls_url: str
    api_key: Union[str, Tuple[str, str]]

    upload_table: Optional[str] = None
    new_task_table: Optional[str] = None
    new_annotation_table: Optional[str] = None
    new_sync_table: Optional[str] = None
    old_step_name: Optional[str] = None


def _compute_transform_meta_table_name(step_name: str, input_table: str, output_table: str) -> str:
    parts = ["BatchTransformStep", step_name, input_table, output_table]
    hasher = hashlib.shake_128()
    hasher.update("".join(parts).encode("utf-8"))
    return f"{step_name}_{hasher.hexdigest(5)}_meta"


def _quote_identifier(dbconn: DBConn, name: str) -> str:
    return dbconn.con.dialect.identifier_preparer.quote(name)


def _rename_table(dbconn: DBConn, old_name: str, new_name: str) -> None:
    old_quoted = _quote_identifier(dbconn, old_name)
    new_quoted = _quote_identifier(dbconn, new_name)
    with dbconn.con.begin() as con:
        con.execute(text(f"ALTER TABLE {old_quoted} RENAME TO {new_quoted}"))


def _get_project(
    ls_url: str, api_key: Union[str, Tuple[str, str]], project_identifier: Union[str, int]
) -> Tuple[LabelStudio, int]:
    resolved_key = api_key if isinstance(api_key, str) else login_and_get_token(ls_url, api_key[0], api_key[1])
    client = LabelStudio(
        base_url=ls_url,
        api_key=resolved_key,
    )
    project: Optional[ProjectDict]
    if str(project_identifier).isnumeric():
        project = project_to_dict(client.projects.get(id=int(project_identifier)))
    else:
        project = get_project_by_title(client, str(project_identifier))
    if project is None:
        raise ValueError(f"Project with identifier={project_identifier!r} not found.")
    assert project is not None
    return client, project["id"]


def _get_input_table_columns(dbconn: DBConn, input_table: str) -> Dict[str, Column[Any]]:
    metadata = MetaData()
    table = Table(input_table, metadata, autoload_with=dbconn.con)
    return {column.name: column for column in table.columns}


def _ensure_columns(
    dbconn: DBConn,
    table_name: str,
    expected_columns: List[Column[Any]],
) -> bool:
    inspector = inspect(dbconn.con)
    if not inspector.has_table(table_name):
        return False
    existing = {col["name"] for col in inspector.get_columns(table_name)}
    missing = [col for col in expected_columns if col.name not in existing]
    for column in missing:
        col_type = column.type.compile(dialect=dbconn.con.dialect)
        quoted = _quote_identifier(dbconn, table_name)
        with dbconn.con.begin() as con:
            con.execute(text(f"ALTER TABLE {quoted} ADD COLUMN {column.name} {col_type}"))
    return len(missing) > 0


def _cleanup_annotations(values: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for ann in values:
        ann = dict(ann)
        if "created_ago" in ann:
            del ann["created_ago"]
        cleaned.append(ann)
    return cleaned


def _group_rows_by_updated_at(
    rows: List[Dict[str, Any]],
) -> Dict[datetime, List[Dict[str, Any]]]:
    grouped: Dict[datetime, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        updated_at = row.pop("_updated_at")
        grouped[updated_at].append(row)
    return grouped


def _parse_updated_at(value: str) -> datetime:
    parsed = datetime.strptime(value, DATETIME_FORMAT)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def migrate_labelstudio_step_to_upload_tasks(
    spec: LabelStudioMigrationSpec,
    dbconn: DBConn,
    input_dbconn: Optional[DBConn] = None,
    rename_transform_meta: bool = True,
    backfill_new_tables: bool = True,
) -> None:
    input_dbconn = input_dbconn or dbconn
    upload_table = spec.upload_table or f"{spec.input_table}_upload"
    new_task_table = spec.new_task_table or upload_table
    new_annotation_table = spec.new_annotation_table or spec.output_table
    new_sync_table = spec.new_sync_table or spec.sync_table
    old_step_name = spec.old_step_name or "upload_data_to_ls"

    input_columns = _get_input_table_columns(input_dbconn, spec.input_table)
    primary_schema: List[Column[Any]] = [
        Column(name, input_columns[name].type, primary_key=True) for name in spec.primary_keys
    ]

    task_schema: List[Column[Any]] = primary_schema + [Column("task_id", Integer)]
    annotation_schema: List[Column[Any]] = primary_schema + [Column("annotations", JSON)]
    sync_schema: List[Column[Any]] = [
        Column("project_id", Integer, primary_key=True),
        Column("last_updated_at", DateTime),
    ]

    task_table_missing = not inspect(dbconn.con).has_table(new_task_table)
    annotation_table_missing = not inspect(dbconn.con).has_table(new_annotation_table)
    sync_table_missing = not inspect(dbconn.con).has_table(new_sync_table)

    if task_table_missing:
        TableStoreDB(
            dbconn=dbconn,
            name=new_task_table,
            data_sql_schema=task_schema,
            create_table=True,
        )
    task_columns_added = _ensure_columns(dbconn, new_task_table, task_schema)

    if annotation_table_missing:
        TableStoreDB(
            dbconn=dbconn,
            name=new_annotation_table,
            data_sql_schema=annotation_schema,
            create_table=True,
        )
    annotation_columns_added = _ensure_columns(dbconn, new_annotation_table, annotation_schema)

    if sync_table_missing:
        TableStoreDB(
            dbconn=dbconn,
            name=new_sync_table,
            data_sql_schema=sync_schema,
            create_table=True,
        )
    _ensure_columns(dbconn, new_sync_table, sync_schema)

    if rename_transform_meta:
        old_meta_name = _compute_transform_meta_table_name(old_step_name, spec.input_table, upload_table)
        new_meta_name = _compute_transform_meta_table_name(
            "upload_tasks_to_label_studio", spec.input_table, new_task_table
        )
        inspector = inspect(dbconn.con)
        if inspector.has_table(old_meta_name) and not inspector.has_table(new_meta_name):
            logger.info("Renaming transform meta table %s -> %s", old_meta_name, new_meta_name)
            _rename_table(dbconn, old_meta_name, new_meta_name)
        elif inspector.has_table(old_meta_name) and inspector.has_table(new_meta_name):
            logger.info(
                "Transform meta table already exists: %s (old=%s)",
                new_meta_name,
                old_meta_name,
            )

    should_backfill = backfill_new_tables and (
        task_table_missing or annotation_table_missing or task_columns_added or annotation_columns_added
    )

    if not should_backfill:
        return

    client, project_id = _get_project(spec.ls_url, spec.api_key, spec.project_identifier)
    ds = DataStore(dbconn, create_meta_table=True)
    dt_task = ds.get_or_create_table(
        new_task_table,
        TableStoreDB(
            dbconn=dbconn,
            name=new_task_table,
            data_sql_schema=task_schema,
            create_table=False,
        ),
    )
    dt_annotation = ds.get_or_create_table(
        new_annotation_table,
        TableStoreDB(
            dbconn=dbconn,
            name=new_annotation_table,
            data_sql_schema=annotation_schema,
            create_table=False,
        ),
    )
    dt_sync = ds.get_or_create_table(
        new_sync_table,
        TableStoreDB(
            dbconn=dbconn,
            name=new_sync_table,
            data_sql_schema=sync_schema,
            create_table=False,
        ),
    )

    max_updated_at: Optional[datetime] = None
    for tasks_page in get_tasks_iter(client, project_id):
        if not tasks_page:
            continue
        task_rows: List[Dict[str, Any]] = []
        annotation_rows: List[Dict[str, Any]] = []
        for task in tasks_page:
            updated_at = _parse_updated_at(task["updated_at"])
            max_updated_at = updated_at if max_updated_at is None else max(max_updated_at, updated_at)
            task_rows.append(
                {
                    **{pk: task["data"].get(pk) for pk in spec.primary_keys},
                    "task_id": task["id"],
                    "_updated_at": updated_at,
                }
            )
            annotation_rows.append(
                {
                    **{pk: task["data"].get(pk) for pk in spec.primary_keys},
                    "annotations": _cleanup_annotations(task.get("annotations", [])),
                    "_updated_at": updated_at,
                }
            )

        for updated_at, rows in _group_rows_by_updated_at(task_rows).items():
            df = pd.DataFrame.from_records(rows)
            if not df.empty:
                df = df.dropna(subset=spec.primary_keys)
                dt_task.store_chunk(df, now=updated_at.timestamp())

        for updated_at, rows in _group_rows_by_updated_at(annotation_rows).items():
            df = pd.DataFrame.from_records(rows)
            if not df.empty:
                df = df.dropna(subset=spec.primary_keys)
                dt_annotation.store_chunk(df, now=updated_at.timestamp())

    if max_updated_at is not None:
        df_sync = pd.DataFrame({"project_id": [project_id], "last_updated_at": [max_updated_at]})
        dt_sync.store_chunk(df_sync, now=max_updated_at.timestamp())
