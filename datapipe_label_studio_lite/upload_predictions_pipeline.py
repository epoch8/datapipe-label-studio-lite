from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import label_studio_sdk
import pandas as pd
import requests
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
from datapipe_label_studio_lite.sdk_utils import get_project_by_title
from datapipe_label_studio_lite.upload_tasks_pipeline import logger
from datapipe_label_studio_lite.utils import check_columns_are_in_table
from sqlalchemy import JSON, Column, Integer, String


def upload_prediction_to_label_studio(
    df__item__has__prediction: pd.DataFrame,
    df__label_studio_project_task: pd.DataFrame,
    idx: IndexDF,
    get_project: Callable[[], label_studio_sdk.Project],
    primary_keys: List[str],
    dt__output__label_studio_project_prediction: DataTable,
    model_version__separator: str,
) -> pd.DataFrame:
    """
    Добавляет в LS предсказания.
    """
    df = pd.merge(df__item__has__prediction, df__label_studio_project_task, on=primary_keys)
    if (df__item__has__prediction.empty and df__label_studio_project_task.empty) and idx.empty:
        return pd.DataFrame(columns=primary_keys + ["task_id", "prediction"])

    project = get_project()
    idx = data_to_index(idx, primary_keys)
    df_existing_prediction_to_be_deleted = dt__output__label_studio_project_prediction.get_data(idx=idx)
    if len(df_existing_prediction_to_be_deleted) > 0:
        for prediction in df_existing_prediction_to_be_deleted["prediction"]:
            try:
                project.make_request(method="DELETE", url=f"api/predictions/{prediction['id']}/")
            except requests.exceptions.HTTPError:
                continue
        dt__output__label_studio_project_prediction.delete_by_idx(
            idx=data_to_index(df_existing_prediction_to_be_deleted, primary_keys)
        )

    if df.empty:
        return pd.DataFrame(columns=primary_keys + ["task_id"])

    df["model_version"] = df.apply(
        lambda row: model_version__separator.join([str(row[column]) for column in primary_keys]),
        axis=1,
    )
    # Не подходит из-за https://github.com/HumanSignal/label-studio/issues/4819
    # uploaded_predictions = self.project.create_predictions(
    #     [
    #         dict(
    #             task=row["task_id"],
    #             result=row["prediction"].get('result', []),
    #             model_version=row['model_version'],
    #             score=row["prediction"].get('score', 1.0)
    #         )
    #         for _, row in df.iterrows()
    #     ]
    # )
    uploaded_predictions = [
        project.create_prediction(
            task_id=row["task_id"],
            result=row["prediction"].get("result", []),
            model_version=row["model_version"],
            score=row["prediction"].get("score", 1.0),
        )
        for _, row in df.iterrows()
    ]
    df["prediction_id"] = [prediction["id"] for prediction in uploaded_predictions]
    df["prediction"] = [prediction for prediction in uploaded_predictions]
    return df[primary_keys + ["task_id", "prediction_id", "model_version", "prediction"]]


@dataclass
class LabelStudioUploadPredictions(PipelineStep):
    input__item__has__prediction: str
    # prediction имеет такой вид: {"result": [...], "score": 0.}
    input__label_studio_project_task: str
    output__label_studio_project_prediction: str

    ls_url: str
    api_key: str
    project_identifier: Union[str, int]  # project_title or id
    primary_keys: List[str]

    chunk_size: int = 100
    create_table: bool = False
    labels: Optional[Labels] = None
    model_version__separator: str = "__"
    executor_config: Optional[ExecutorConfig] = None

    def __post_init__(self):
        if isinstance(self.project_identifier, str):
            assert len(self.project_identifier) <= 50

        # lazy initialization
        self._ls_client: Optional[label_studio_sdk.Client] = None
        self._project: Optional[label_studio_sdk.Project] = None
        self.labels = self.labels or []

    @property
    def ls_client(self) -> label_studio_sdk.Client:
        if self._ls_client is None:
            self._ls_client = label_studio_sdk.Client(
                url=self.ls_url,
                api_key=self.api_key if isinstance(self.api_key, str) else None,
                credentials=self.api_key if isinstance(self.api_key, tuple) else None,
            )
        return self._ls_client

    def get_project(self) -> label_studio_sdk.Project:
        """
        При первом использовании ищет проект в LS по индентификатору,
        если его нет -- автоматически создаётся проект с нуля.
        """
        if self._project is not None:
            return self._project
        assert self.ls_client.check_connection(), "No connection to LS."
        self._project = (
            self.ls_client.get_project(int(self.project_identifier))
            if str(self.project_identifier).isnumeric()
            else get_project_by_title(self.ls_client, str(self.project_identifier))
        )
        if self._project is None:
            raise ValueError(f"Project with {self.project_identifier=} not found")
        return self._project

    def build_compute(self, ds: DataStore, catalog: Catalog) -> List[ComputeStep]:
        dt__input__has__prediction = ds.get_table(self.input__item__has__prediction)
        assert isinstance(dt__input__has__prediction.table_store, TableStoreDB)
        check_columns_are_in_table(ds, self.input__item__has__prediction, self.primary_keys + ["prediction"])
        check_columns_are_in_table(ds, self.input__label_studio_project_task, self.primary_keys + ["task_id"])
        catalog.add_datatable(
            self.output__label_studio_project_prediction,
            Table(
                ds.get_or_create_table(
                    self.output__label_studio_project_prediction,
                    TableStoreDB(
                        dbconn=ds.meta_dbconn,
                        name=self.output__label_studio_project_prediction,
                        data_sql_schema=[
                            column
                            for column in dt__input__has__prediction.primary_schema
                            if column.name in self.primary_keys
                        ]
                        + [
                            Column("task_id", Integer),
                            Column("prediction_id", Integer),
                            Column("model_version", String),
                            Column("prediction", JSON),
                        ],
                        create_table=self.create_table,
                    ),
                ).table_store
            ),
        )
        dt__output__label_studio_project_prediction = ds.get_table(self.output__label_studio_project_prediction)

        pipeline = Pipeline(
            [
                BatchTransform(
                    labels=self.labels,
                    func=upload_prediction_to_label_studio,
                    inputs=[self.input__item__has__prediction, self.input__label_studio_project_task],
                    outputs=[self.output__label_studio_project_prediction],
                    chunk_size=self.chunk_size,
                    executor_config=self.executor_config,
                    kwargs=dict(
                        get_project=self.get_project,
                        primary_keys=self.primary_keys,
                        dt__output__label_studio_project_prediction=dt__output__label_studio_project_prediction,
                        model_version__separator=self.model_version__separator,
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
    ls_client: label_studio_sdk.Client,
    primary_keys: List[str],
    dt__output__label_studio_project_prediction: DataTable,
    model_version__separator: str,
) -> pd.DataFrame:
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
        df__res = upload_prediction_to_label_studio(
            df__item__has__prediction=df__item__has__prediction_by_project_identifier,
            df__label_studio_project_task=df__label_studio_project_task_by_project_identifier,
            idx=idx_by_project_identifier,
            get_project=lambda: ls_client.get_project(project_id),
            primary_keys=primary_keys,
            dt__output__label_studio_project_prediction=dt__output__label_studio_project_prediction,
            model_version__separator=model_version__separator,
        )
        dfs.append(df__res)
    if len(dfs) == 0:
        dfs_res = pd.DataFrame(columns=primary_keys + ["task_id", "prediction_id", "model_version", "prediction"])
    else:
        dfs_res = pd.concat(dfs, ignore_index=True)
    print(f"{dfs_res=}")
    return dfs_res


@dataclass
class LabelStudioUploadPredictionsToProjects(PipelineStep):
    input__item__has__prediction: str
    # prediction имеет такой вид: {"result": [...], "score": 0.}
    input__label_studio_project: str
    input__label_studio_project_task: str
    output__label_studio_project_prediction: str

    ls_url: str
    api_key: str
    primary_keys: List[str]

    chunk_size: int = 100
    create_table: bool = False
    labels: Optional[Labels] = None
    model_version__separator: str = "__"
    executor_config: Optional[ExecutorConfig] = None

    def __post_init__(self):
        # lazy initialization
        self._ls_client: Optional[label_studio_sdk.Client] = None
        self._project: Optional[label_studio_sdk.Project] = None
        self.labels = self.labels or []

    @property
    def ls_client(self) -> label_studio_sdk.Client:
        if self._ls_client is None:
            self._ls_client = label_studio_sdk.Client(
                url=self.ls_url,
                api_key=self.api_key if isinstance(self.api_key, str) else None,
                credentials=self.api_key if isinstance(self.api_key, tuple) else None,
            )
        return self._ls_client

    def build_compute(self, ds: DataStore, catalog: Catalog) -> List[ComputeStep]:
        assert "project_identifier" in self.primary_keys
        dt__input__has__prediction = ds.get_table(self.input__item__has__prediction)
        dt__input__label_studio_project = ds.get_table(self.input__label_studio_project)
        assert isinstance(dt__input__has__prediction.table_store, TableStoreDB)
        check_columns_are_in_table(ds, self.input__item__has__prediction, self.primary_keys + ["prediction"])
        check_columns_are_in_table(ds, self.input__label_studio_project_task, self.primary_keys + ["task_id"])
        check_columns_are_in_table(ds, self.input__label_studio_project, ["project_identifier", "project_id"])
        catalog.add_datatable(
            self.output__label_studio_project_prediction,
            Table(
                ds.get_or_create_table(
                    self.output__label_studio_project_prediction,
                    TableStoreDB(
                        dbconn=ds.meta_dbconn,
                        name=self.output__label_studio_project_prediction,
                        data_sql_schema=[
                            column
                            for column in dt__input__has__prediction.primary_schema
                            if column.name in self.primary_keys
                        ]
                        + [
                            Column("task_id", Integer),
                            Column("prediction_id", Integer),
                            Column("model_version", String),
                            Column("prediction", JSON),
                        ],
                        create_table=self.create_table,
                    ),
                ).table_store
            ),
        )
        dt__output__label_studio_project_prediction = ds.get_table(self.output__label_studio_project_prediction)

        pipeline = Pipeline(
            [
                BatchTransform(
                    labels=[("stage", "upload_predictions_to_ls"), *(self.labels or [])],
                    func=upload_prediction_to_label_studio_projects,
                    inputs=[
                        self.input__label_studio_project,
                        self.input__item__has__prediction,
                        self.input__label_studio_project_task,
                    ],
                    outputs=[self.output__label_studio_project_prediction],
                    chunk_size=self.chunk_size,
                    executor_config=self.executor_config,
                    kwargs=dict(
                        ls_client=self.ls_client,
                        primary_keys=self.primary_keys,
                        dt__output__label_studio_project_prediction=dt__output__label_studio_project_prediction,
                        model_version__separator=self.model_version__separator,
                    ),
                    transform_keys=self.primary_keys,
                ),
            ]
        )
        return build_compute(ds, catalog, pipeline)
