import logging
from datapipe_label_studio_lite.utils import check_columns_are_in_table
import pandas as pd
from typing import Any, Dict, Union, List, Optional, cast
from dataclasses import dataclass
from datapipe.store.database import TableStoreDB

from sqlalchemy import Integer, Column
from datapipe.executor import ExecutorConfig

from datapipe.types import (
    Labels,
)
from datapipe.compute import (
    Pipeline,
    PipelineStep,
    DataStore,
    Catalog,
    build_compute,
)
from datapipe.step.batch_transform import BatchTransform
from datapipe.step.datatable_transform import DatatableTransformStep
import label_studio_sdk
from datapipe_label_studio_lite.sdk_utils import get_project_by_title


logger = logging.getLogger("dataipipe_label_studio_lite")


@dataclass
class GCSBucket:
    bucket: str
    google_application_credentials: Optional[str] = None

    @property
    def type(self):
        return "gcs"


@dataclass
class S3Bucket:
    bucket: str
    key: str
    secret: str
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None

    @property
    def type(self):
        return "s3"


@dataclass
class Buckets:
    buckets: List[Union[GCSBucket, S3Bucket]]


@dataclass
class CreateLabelStudioProjects(PipelineStep):
    input__label_studio_project_setting: str  # Input with columns: 'project_label_config', 'project_description', 'storages'
    output__label_studio_project: str

    ls_url: str
    api_key: str
    primary_keys: List[str]

    storages: Optional[List[Union[GCSBucket, S3Bucket]]] = None
    create_table: bool = False
    labels: Optional[Labels] = None
    executor_config: Optional[ExecutorConfig] = None

    def __post_init__(self):
        # lazy initialization
        self._ls_client: Optional[label_studio_sdk.Client] = None
        self.labels = self.labels or []
        self.storages = self.storages or []

    @property
    def ls_client(self) -> label_studio_sdk.Client:
        if self._ls_client is None:
            self._ls_client = label_studio_sdk.Client(
                url=self.ls_url,
                api_key=self.api_key if isinstance(self.api_key, str) else None,
                credentials=self.api_key if isinstance(self.api_key, tuple) else None,
            )
        return self._ls_client

    def create_project(
        self,
        project_identifier: Union[str, int],  # project_title or id
        project_label_config_at_create: Dict[str, Any],
        project_description_at_create: Dict[str, Any],
    ) -> label_studio_sdk.Project:
        """
        При первом использовании ищет проект в LS по индентификатору,
        если его нет -- автоматически создаётся проект с нуля.
        """
        if isinstance(project_identifier, str):
            assert len(project_identifier) <= 50
        assert self.ls_client.check_connection(), "No connection to LS."
        project = (
            self.ls_client.get_project(int(project_identifier))
            if str(project_identifier).isnumeric()
            else get_project_by_title(self.ls_client, str(project_identifier))
        )
        if project is None:
            project = self.ls_client.start_project(
                title=project_identifier,
                description=project_description_at_create,
                label_config=project_label_config_at_create,
                expert_instruction="",
                show_instruction=False,
                show_skip_button=False,
                enable_empty_annotation=True,
                show_annotation_history=False,
                organization=1,
                color="#FFFFFF",
                maximum_annotations=1,
                is_published=False,
                model_version="",
                is_draft=False,
                min_annotations_to_start_training=10,
                show_collab_predictions=True,
                sampling="Sequential sampling",
                show_ground_truth_first=True,
                show_overlap_first=True,
                overlap_cohort_percentage=100,
                task_data_login=None,
                task_data_password=None,
                control_weights={},
            )
            logger.info(f"Project with {project_identifier=} not found, created new project with id={project.id}")
        storages_response = self.ls_client.make_request("GET", "/api/storages", params=dict(project=project.id))
        connected_buckets = [
            f"{storage['type']}://{storage.get('bucket', None)}" for storage in storages_response.json()
        ]
        for storage in cast(List[Union[GCSBucket, S3Bucket]], self.storages):
            if (storage_name := f"{storage.type}://{storage.bucket}") not in connected_buckets:
                if isinstance(storage, S3Bucket):
                    result = project.connect_s3_import_storage(
                        bucket=storage.bucket,
                        title=storage_name,
                        aws_access_key_id=storage.key,
                        aws_secret_access_key=storage.secret,
                        s3_endpoint=storage.endpoint_url,
                        region_name=storage.region_name,
                    )
                elif isinstance(storage, GCSBucket):
                    result = project.connect_google_import_storage(
                        bucket=storage.bucket,
                        title=storage_name,
                        google_application_credentials=storage.google_application_credentials,
                    )
                logger.info(f"Adding storage {storage_name=} to project: {result}")
        return project

    def build_compute(self, ds: DataStore, catalog: Catalog) -> List[DatatableTransformStep]:
        dt__input__label_studio_project_setting = ds.get_table(self.input__label_studio_project_setting)
        assert isinstance(dt__input__label_studio_project_setting.table_store, TableStoreDB)
        check_columns_are_in_table(
            ds,
            self.input__label_studio_project_setting,
            ["project_identifier", "project_label_config_at_create", "project_description_at_create"],
        )
        catalog.add_datatable(
            self.output__label_studio_project,
            ds.get_or_create_table(
                self.output__label_studio_project,
                TableStoreDB(
                    dbconn=ds.meta_dbconn,
                    name=self.output__label_studio_project,
                    data_sql_schema=[
                        column
                        for column in dt__input__label_studio_project_setting.table_store.get_primary_schema()
                        if column.name in ["project_identifier"]
                    ]
                    + [
                        Column("project_id", Integer, primary_key=True),
                    ],
                    create_table=self.create_table,
                ),
            ),
        )

        def create_projects(df__input__label_studio_project_setting) -> pd.DataFrame:
            """
            Добавляет в LS новые задачи с заданными ключами.
            """
            project_identifier = df__input__label_studio_project_setting.iloc[0]["project_identifier"]
            project_label_config_at_create = df__input__label_studio_project_setting.iloc[0][
                "project_label_config_at_create"
            ]
            project_description_at_create = df__input__label_studio_project_setting.iloc[0][
                "project_description_at_create"
            ]
            project = self.create_project(
                project_identifier, project_label_config_at_create, project_description_at_create
            )
            df__input__label_studio_project_setting["project_id"] = project.id
            return df__input__label_studio_project_setting

        pipeline = Pipeline(
            [
                BatchTransform(
                    labels=self.labels,
                    func=create_projects,
                    inputs=[self.input__label_studio_project_setting],
                    outputs=[self.output__label_studio_project],
                    chunk_size=1,
                    executor_config=self.executor_config,
                ),
            ]
        )
        return build_compute(ds, catalog, pipeline)
