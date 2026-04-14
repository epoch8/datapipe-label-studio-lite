import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from datapipe.compute import (
    Catalog,
    ComputeStep,
    DataStore,
    Pipeline,
    PipelineStep,
    Table,
    build_compute,
)
from datapipe.datatable import DataTable
from datapipe.executor import ExecutorConfig
from datapipe.step.batch_transform import BatchTransform
from datapipe.store.database import TableStoreDB
from datapipe.types import IndexDF, Labels, data_to_index
from label_studio_sdk import LabelStudio
from sqlalchemy import JSON, Column, Integer, String

from datapipe_label_studio_lite.sdk_utils import (
    get_ls_client,
    resolve_project_id,
)
from datapipe_label_studio_lite.upload_tasks_pipeline import logger
from datapipe_label_studio_lite.utils import check_columns_are_in_table

_VALID_UPLOAD_MODES = {"prediction", "annotation"}


def _make_jsonable(value: object) -> object:
    return json.loads(json.dumps(value, default=str))


def _normalize_optional_value(value: object) -> Optional[object]:
    if pd.isna(value):
        return None
    return value


def _get_upload_columns(
    upload_mode: str,
    model_version__column: str,
    annotations_completed_by__column: str,
) -> Tuple[str, str, str, str]:
    if upload_mode not in _VALID_UPLOAD_MODES:
        raise ValueError(
            f"Unsupported upload_mode='{upload_mode}'. "
            f"Expected one of {sorted(_VALID_UPLOAD_MODES)}."
        )
    if upload_mode == "prediction":
        return ("prediction", "prediction_id", "model_version", model_version__column)
    return ("annotation", "annotation_id", "completed_by", annotations_completed_by__column)


def upload_prediction_to_label_studio(
    df__item__has__prediction: pd.DataFrame,
    df__label_studio_project_task: pd.DataFrame,
    idx: IndexDF,
    get_project_context: Callable[[], Tuple[LabelStudio, int]],
    primary_keys: List[str],
    dt__output__label_studio_project_result: DataTable,
    model_version__column: str,
    annotations_completed_by__column: str,
    upload_mode: str = "prediction",
) -> pd.DataFrame:
    """
    Добавляет в LS предсказания или аннотации.
    """
    (
        item_column,
        item_id_column,
        item_meta_column,
        item_meta_input_column,
    ) = _get_upload_columns(upload_mode, model_version__column, annotations_completed_by__column)

    df = pd.merge(df__item__has__prediction, df__label_studio_project_task, on=primary_keys)
    if (df__item__has__prediction.empty and df__label_studio_project_task.empty) and idx.empty:
        return pd.DataFrame(columns=primary_keys + ["task_id", item_column])

    client, project_id = get_project_context()
    idx = data_to_index(idx, primary_keys)
    df_existing_item_to_be_deleted = dt__output__label_studio_project_result.get_data(idx=idx)
    if len(df_existing_item_to_be_deleted) > 0:
        for item in df_existing_item_to_be_deleted[item_column]:
            item_id = item.get("id") if isinstance(item, dict) else None
            if item_id is None:
                continue
            try:
                if upload_mode == "prediction":
                    client.predictions.delete(id=item_id)
                else:
                    client.annotations.delete(id=item_id)
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code not in [404, 500]:
                    raise
        dt__output__label_studio_project_result.delete_by_idx(
            idx=data_to_index(df_existing_item_to_be_deleted, primary_keys)
        )

    if df.empty:
        return pd.DataFrame(columns=primary_keys + ["task_id"])

    df[item_meta_column] = df[item_meta_input_column]
    uploaded_items = []
    for _, row in df.iterrows():
        payload = row[item_column]
        if not isinstance(payload, dict):
            payload = {}

        if upload_mode == "prediction":
            item = client.predictions.create(
                task=row["task_id"],
                result=payload.get("result", []),
                model_version=row[item_meta_column],
                score=payload.get("score", 1.0),
            )
        else:
            kwargs = {
                "task": row["task_id"],
                "result": payload.get("result", []),
                "was_cancelled": False,
            }
            completed_by = _normalize_optional_value(row[item_meta_column])
            if completed_by is not None:
                kwargs["completed_by"] = completed_by
            item = client.annotations.create(**kwargs)
        uploaded_items.append(item)
    df[item_id_column] = [getattr(item, "id", None) for item in uploaded_items]
    df[item_column] = pd.Series(
        [_make_jsonable(item.model_dump() if hasattr(item, "model_dump") else item) for item in uploaded_items],
        dtype=object,
        index=df.index,
    )
    return df[primary_keys + ["task_id", item_id_column, item_meta_column, item_column]]


@dataclass
class LabelStudioUploadPredictions(PipelineStep):
    input__item__has__result: str
    # prediction/annotation имеет такой вид: {"result": [...], "score": 0.}
    input__label_studio_project_task: str
    output__label_studio_project_result: str

    ls_url: str
    api_key: str
    project_identifier: Union[str, int]  # project_title or id
    primary_keys: List[str]

    chunk_size: int = 100
    create_table: bool = False
    labels: Optional[Labels] = None
    model_version__column: str = "model_version"
    annotations_completed_by__column: str = "annotations_completed_by"
    upload_mode: str = "prediction"
    executor_config: Optional[ExecutorConfig] = None

    def __post_init__(self):
        if isinstance(self.project_identifier, str):
            assert len(self.project_identifier) <= 50

        # lazy initialization
        self._ls_client: Optional[LabelStudio] = None
        self._project_id: Optional[int] = None
        self.labels = self.labels or []
        _get_upload_columns(
            self.upload_mode,
            self.model_version__column,
            self.annotations_completed_by__column,
        )

    @property
    def ls_client(self) -> LabelStudio:
        if self._ls_client is None:
            self._ls_client = get_ls_client(self.ls_url, self.api_key)
        return self._ls_client

    def get_project_context(self) -> Tuple[LabelStudio, int]:
        """
        При первом использовании ищет проект в LS по индентификатору,
        если его нет -- автоматически создаётся проект с нуля.
        """
        if self._project_id is not None:
            return (self.ls_client, self._project_id)
        self._project_id = resolve_project_id(self.ls_client, self.project_identifier)
        return (self.ls_client, self._project_id)

    def build_compute(self, ds: DataStore, catalog: Catalog) -> List[ComputeStep]:
        (
            item_column,
            item_id_column,
            item_meta_column,
            item_meta_input_column,
        ) = _get_upload_columns(
            self.upload_mode,
            self.model_version__column,
            self.annotations_completed_by__column,
        )
        dt__input__has__prediction = ds.get_table(self.input__item__has__result)
        assert isinstance(dt__input__has__prediction.table_store, TableStoreDB)
        check_columns_are_in_table(
            ds,
            self.input__item__has__result,
            self.primary_keys + [item_column, item_meta_input_column],
        )
        check_columns_are_in_table(ds, self.input__label_studio_project_task, self.primary_keys + ["task_id"])
        catalog.add_datatable(
            self.output__label_studio_project_result,
            Table(
                ds.get_or_create_table(
                    self.output__label_studio_project_result,
                    TableStoreDB(
                        dbconn=ds.meta_dbconn,
                        name=self.output__label_studio_project_result,
                        data_sql_schema=[
                            column
                            for column in dt__input__has__prediction.primary_schema
                            if column.name in self.primary_keys
                        ]
                        + [
                            Column("task_id", Integer),
                            Column(item_id_column, Integer),
                            Column(item_meta_column, String),
                            Column(item_column, JSON),
                        ],
                        create_table=self.create_table,
                    ),
                ).table_store
            ),
        )
        dt__output__label_studio_project_result = ds.get_table(self.output__label_studio_project_result)

        pipeline = Pipeline(
            [
                BatchTransform(
                    labels=self.labels,
                    func=upload_prediction_to_label_studio,
                    inputs=[self.input__item__has__result, self.input__label_studio_project_task],
                    outputs=[self.output__label_studio_project_result],
                    chunk_size=self.chunk_size,
                    executor_config=self.executor_config,
                    kwargs=dict(
                        get_project_context=self.get_project_context,
                        primary_keys=self.primary_keys,
                        dt__output__label_studio_project_result=dt__output__label_studio_project_result,
                        model_version__column=self.model_version__column,
                        annotations_completed_by__column=self.annotations_completed_by__column,
                        upload_mode=self.upload_mode,
                    ),
                ),
            ]
        )
        return build_compute(ds, catalog, pipeline)


def upload_prediction_to_label_studio_projects(
    df__label_studio_project: pd.DataFrame,
    df__item__has__prediction: pd.DataFrame,
    df__label_studio_project_task: pd.DataFrame,
    idx: IndexDF,
    ls_client: LabelStudio,
    primary_keys: List[str],
    dt__output__label_studio_project_result: DataTable,
    model_version__column: str,
    annotations_completed_by__column: str,
    upload_mode: str = "prediction",
) -> pd.DataFrame:
    (
        item_column,
        item_id_column,
        item_meta_column,
        _,
    ) = _get_upload_columns(upload_mode, model_version__column, annotations_completed_by__column)

    project_identifiers = (
        set(df__label_studio_project["project_identifier"])
        .union(set(df__item__has__prediction["project_identifier"]))
        .union(set(df__label_studio_project_task["project_identifier"]))
        .union(set(idx["project_identifier"]))
    )
    dfs = []
    for project_identifier in project_identifiers:
        if project_identifier not in set(df__label_studio_project["project_identifier"]):
            logger.info(f"Project {project_identifier} not found in input__label_studio_project. Skipping")
            continue
        project_id = df__label_studio_project[
            df__label_studio_project["project_identifier"] == project_identifier
        ].iloc[0]["project_id"]
        df__item__has__prediction_by_project_identifier = df__item__has__prediction[
            df__item__has__prediction["project_identifier"] == project_identifier
        ]
        df__label_studio_project_task_by_project_identifier = df__label_studio_project_task[
            df__label_studio_project_task["project_identifier"] == project_identifier
        ]
        idx_by_project_identifier = idx[idx["project_identifier"] == project_identifier]

        def _get_project_context(project_id: int = project_id) -> Tuple[LabelStudio, int]:
            return (ls_client, project_id)

        df__res = upload_prediction_to_label_studio(
            df__item__has__prediction=df__item__has__prediction_by_project_identifier,
            df__label_studio_project_task=df__label_studio_project_task_by_project_identifier,
            idx=idx_by_project_identifier,
            get_project_context=_get_project_context,
            primary_keys=primary_keys,
            dt__output__label_studio_project_result=dt__output__label_studio_project_result,
            model_version__column=model_version__column,
            annotations_completed_by__column=annotations_completed_by__column,
            upload_mode=upload_mode,
        )
        dfs.append(df__res)
    if len(dfs) == 0:
        dfs_res = pd.DataFrame(
            columns=primary_keys + ["task_id", item_id_column, item_meta_column, item_column]
        )
    else:
        dfs_res = pd.concat(dfs, ignore_index=True)
    return dfs_res


@dataclass
class LabelStudioUploadPredictionsToProjects(PipelineStep):
    input__item__has__prediction: str
    # prediction/annotation имеет такой вид: {"result": [...], "score": 0.}
    input__label_studio_project: str
    input__label_studio_project_task: str
    output__label_studio_project_result: str

    ls_url: str
    api_key: str
    primary_keys: List[str]

    chunk_size: int = 100
    create_table: bool = False
    labels: Optional[Labels] = None
    model_version__column: str = "model_version"
    annotations_completed_by__column: str = "annotations_completed_by"
    upload_mode: str = "prediction"
    executor_config: Optional[ExecutorConfig] = None

    def __post_init__(self):
        # lazy initialization
        self._ls_client: Optional[LabelStudio] = None
        self.labels = self.labels or []
        _get_upload_columns(
            self.upload_mode,
            self.model_version__column,
            self.annotations_completed_by__column,
        )

    @property
    def ls_client(self) -> LabelStudio:
        if self._ls_client is None:
            self._ls_client = get_ls_client(self.ls_url, self.api_key)
        return self._ls_client

    def build_compute(self, ds: DataStore, catalog: Catalog) -> List[ComputeStep]:
        assert "project_identifier" in self.primary_keys
        (
            item_column,
            item_id_column,
            item_meta_column,
            item_meta_input_column,
        ) = _get_upload_columns(
            self.upload_mode,
            self.model_version__column,
            self.annotations_completed_by__column,
        )
        dt__input__has__prediction = ds.get_table(self.input__item__has__prediction)
        assert isinstance(dt__input__has__prediction.table_store, TableStoreDB)
        check_columns_are_in_table(
            ds,
            self.input__item__has__prediction,
            self.primary_keys + [item_column, item_meta_input_column],
        )
        check_columns_are_in_table(ds, self.input__label_studio_project_task, self.primary_keys + ["task_id"])
        check_columns_are_in_table(ds, self.input__label_studio_project, ["project_identifier", "project_id"])
        catalog.add_datatable(
            self.output__label_studio_project_result,
            Table(
                ds.get_or_create_table(
                    self.output__label_studio_project_result,
                    TableStoreDB(
                        dbconn=ds.meta_dbconn,
                        name=self.output__label_studio_project_result,
                        data_sql_schema=[
                            column
                            for column in dt__input__has__prediction.primary_schema
                            if column.name in self.primary_keys
                        ]
                        + [
                            Column("task_id", Integer),
                            Column(item_id_column, Integer),
                            Column(item_meta_column, String),
                            Column(item_column, JSON),
                        ],
                        create_table=self.create_table,
                    ),
                ).table_store
            ),
        )
        dt__output__label_studio_project_result = ds.get_table(self.output__label_studio_project_result)

        pipeline = Pipeline(
            [
                BatchTransform(
                    labels=self.labels,
                    func=upload_prediction_to_label_studio_projects,
                    inputs=[
                        self.input__label_studio_project,
                        self.input__item__has__prediction,
                        self.input__label_studio_project_task,
                    ],
                    outputs=[self.output__label_studio_project_result],
                    chunk_size=self.chunk_size,
                    executor_config=self.executor_config,
                    kwargs=dict(
                        ls_client=self.ls_client,
                        primary_keys=self.primary_keys,
                        dt__output__label_studio_project_result=dt__output__label_studio_project_result,
                        model_version__column=self.model_version__column,
                        annotations_completed_by__column=self.annotations_completed_by__column,
                        upload_mode=self.upload_mode,
                    ),
                    transform_keys=self.primary_keys,
                ),
            ]
        )
        return build_compute(ds, catalog, pipeline)
