from datapipe.executor import ExecutorConfig
from datapipe_label_studio_lite.utils import check_columns_are_in_table
import pandas as pd
from typing import Union, List, Optional
from dataclasses import dataclass
from datapipe.store.database import TableStoreDB
import requests

from sqlalchemy import Integer, Column, JSON, String

from datapipe.types import (
    IndexDF,
    Labels,
    data_to_index,
)
from datapipe.compute import (
    Pipeline,
    PipelineStep,
    DataStore,
    Table,
    Catalog,
    build_compute,
)
from datapipe.step.batch_transform import BatchTransform
from datapipe.step.datatable_transform import DatatableTransformStep
from datapipe.store.database import DBConn
import label_studio_sdk
from datapipe_label_studio_lite.sdk_utils import get_project_by_title


@dataclass
class LabelStudioUploadPredictions(PipelineStep):
    input__item__has__prediction: str
    # prediction имеет такой вид: {"result": [...], "score": 0.}
    input__label_studio_project_task: str
    output__label_studio_project_prediction: str

    ls_url: str
    api_key: str
    project_identifier: Union[str, int]  # project_title or id
    columns: List[str]

    chunk_size: int = 100
    dbconn: Optional[DBConn] = None
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

    @property
    def project(self) -> label_studio_sdk.Project:
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

    def build_compute(self, ds: DataStore, catalog: Catalog) -> List[DatatableTransformStep]:
        dt__input__has__prediction = ds.get_table(self.input__item__has__prediction)
        assert isinstance(dt__input__has__prediction.table_store, TableStoreDB)
        primary_keys = dt__input__has__prediction.table_store.primary_keys
        check_columns_are_in_table(ds, self.input__item__has__prediction, primary_keys + ["prediction"])
        check_columns_are_in_table(ds, self.input__label_studio_project_task, primary_keys + ["task_id"])
        catalog.add_datatable(
            self.output__label_studio_project_prediction,
            Table(
                ds.get_or_create_table(
                    self.output__label_studio_project_prediction,
                    TableStoreDB(
                        dbconn=ds.meta_dbconn,
                        name=self.output__label_studio_project_prediction,
                        data_sql_schema=[column for column in dt__input__has__prediction.primary_schema]
                        + [
                            Column("task_id", Integer),
                            Column("prediction_id", Integer),
                            Column("model_version", String),
                            Column("prediction", JSON),
                        ],
                        create_table=self.create_table,
                    ),
                )
            ),
        )
        dt__output__label_studio_project_prediction = ds.get_table(self.output__label_studio_project_prediction)

        def upload_prediction_to_label_studio(
            df__item__has__prediction: pd.DataFrame, df__label_studio_project_task: pd.DataFrame, idx: IndexDF
        ) -> pd.DataFrame:
            """
            Добавляет в LS предсказания.
            """
            df = pd.merge(df__item__has__prediction, df__label_studio_project_task, on=primary_keys)
            if (df__item__has__prediction.empty and df__label_studio_project_task.empty) and idx.empty:
                return pd.DataFrame(columns=primary_keys + ["task_id", "prediction"])

            idx = data_to_index(idx, primary_keys)
            df_existing_prediction_to_be_deleted = dt__output__label_studio_project_prediction.get_data(idx=idx)
            if len(df_existing_prediction_to_be_deleted) > 0:
                for prediction in df_existing_prediction_to_be_deleted["prediction"]:
                    try:
                        self.project.make_request(method="DELETE", url=f"api/predictions/{prediction['id']}/")
                    except requests.exceptions.HTTPError:
                        continue
                dt__output__label_studio_project_prediction.delete_by_idx(
                    idx=data_to_index(df_existing_prediction_to_be_deleted, primary_keys)
                )

            if df.empty:
                return pd.DataFrame(columns=primary_keys + ["task_id"])

            df["model_version"] = df.apply(
                lambda row: self.model_version__separator.join([str(row[column]) for column in primary_keys]), axis=1
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
                self.project.create_prediction(
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

        pipeline = Pipeline(
            [
                BatchTransform(
                    labels=[("stage", "upload_predictions_to_ls"), *(self.labels or [])],
                    func=upload_prediction_to_label_studio,
                    inputs=[self.input__item__has__prediction, self.input__label_studio_project_task],
                    outputs=[self.output__label_studio_project_prediction],
                    chunk_size=self.chunk_size,
                    executor_config=self.executor_config,
                ),
            ]
        )
        return build_compute(ds, catalog, pipeline)
