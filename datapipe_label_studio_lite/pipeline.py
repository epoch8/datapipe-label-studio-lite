import numpy as np
import pandas as pd
from typing import Any, Union, List, Optional
from datetime import datetime
from dataclasses import dataclass
from datapipe.run_config import RunConfig
from datapipe.store.database import TableStoreDB

from sqlalchemy import Integer, Column, JSON, DateTime, String

from datapipe.types import ChangeList, data_to_index, index_to_data
from datapipe.compute import PipelineStep, DataStore, Catalog, DatatableTransformStep
from datapipe.core_steps import BatchTransformStep, DataTable
from datapipe.store.database import DBConn
import label_studio_sdk
from datapipe_label_studio_lite.sdk_utils import get_project_by_title, get_tasks_iter
from label_studio_sdk.data_manager import Filters, Operator, Type


class DatatableTransformStepNoChangeList(DatatableTransformStep):
    def run_changelist(self, ds: DataStore, changelist: ChangeList, run_config: RunConfig = None) -> ChangeList:
        return ChangeList()


@dataclass
class LabelStudioStep(PipelineStep):
    input: str
    output: str
    sync_table: str

    ls_url: str
    api_key: str
    dbconn: Union[DBConn, str]
    project_identifier: Union[str, int]  # project_title or id
    data_sql_schema: List[Column]

    project_label_config_at_create: str = ''
    project_description_at_create: str = ''

    create_table: bool = False

    def __post_init__(self):
        assert self.dbconn is not None
        self.data_sql_schema: List[Column] = [column for column in self.data_sql_schema]
        self.data_sql_schema_primary: List[Column] = [column for column in self.data_sql_schema if column.primary_key]
        self.data_columns: List[str] = [column.name for column in self.data_sql_schema if not column.primary_key]
        self.primary_keys = [column.name for column in self.data_sql_schema if column.primary_key]
        for column in ['task_id', 'annotations']:
            assert column not in self.data_columns and column not in self.primary_keys, (
                f'The column "{column}" is reserved for this PipelineStep.'
            )
        if isinstance(self.project_identifier, str):
            assert len(self.project_identifier) <= 50

        # lazy initialization
        self._ls_client: Optional[label_studio_sdk.Client] = None
        self._project: Optional[label_studio_sdk.Project] = None

    @property
    def ls_client(self) -> label_studio_sdk.Client:
        if self._ls_client is None:
            self._ls_client = label_studio_sdk.Client(
                url=self.ls_url,
                api_key=self.api_key if isinstance(self.api_key, str) else None,
                credentials=self.api_key if isinstance(self.api_key, tuple) else None
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
            self.project_identifier if str(self.project_identifier).isnumeric()
            else get_project_by_title(self.ls_client, str(self.project_identifier))
        )
        if self._project is None:
            self._project = self.ls_client.start_project(
                title=self.project_identifier,
                description=self.project_description_at_create,
                label_config=self.project_label_config_at_create,
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
                control_weights={}
            )

        return self._project

    def _convert_data_if_need(self, value: Any):
        if isinstance(value, np.int64):
            return int(value)
        return value

    def build_compute(
        self, ds: DataStore, catalog: Catalog
    ) -> List[DatatableTransformStep]:
        input_dt = catalog.get_datatable(ds, self.input)
        output_uploader_dt = ds.get_or_create_table(
            f'{self.output}_upload', TableStoreDB(
                dbconn=self.dbconn,
                name=f'{self.output}_upload',
                data_sql_schema=self.data_sql_schema_primary + [Column('task_id', Integer)],
                create_table=self.create_table
            )
        )
        sync_datetime_dt = ds.get_or_create_table(
            self.sync_table, TableStoreDB(
                dbconn=self.dbconn,
                name=self.sync_table,
                data_sql_schema=[Column('project_id', String, primary_key=True), Column('datetime', DateTime)],
                create_table=self.create_table
            )
        )
        output_dt = ds.get_or_create_table(
            self.output, TableStoreDB(
                dbconn=self.dbconn,
                name=self.output,
                data_sql_schema=self.data_sql_schema_primary + [Column('annotations', JSON)],
                create_table=self.create_table
            )
        )

        def upload_tasks(df: pd.DataFrame):
            """
                Добавляет в LS новые задачи с заданными ключами.
                (Не поддерживает удаление задач, если в input они пропадают)
            """
            if df.empty or len(df) == 0:
                return

            # Удаляем существующие задачи и перезаливаем их
            df_idx = data_to_index(df, self.primary_keys)
            existing_tasks_df_without_annotations = output_uploader_dt.get_data(idx=df_idx)
            if len(existing_tasks_df_without_annotations) > 0:
                existing_idx = data_to_index(existing_tasks_df_without_annotations, self.primary_keys)
                df_to_be_deleted = pd.merge(
                    left=index_to_data(df, existing_idx),
                    right=existing_tasks_df_without_annotations[self.primary_keys + ['task_id']],
                    on=self.primary_keys
                )
                for task_id in df_to_be_deleted['task_id']:
                    self.project.make_request(
                        method='DELETE', url=f"api/tasks/{task_id}/",
                    )

            # Добавляем новые задачи
            data_to_be_added = [
                {
                    'data': {
                        **{
                            primary_key: self._convert_data_if_need(df.loc[idx, primary_key])
                            for primary_key in self.primary_keys + self.data_columns
                        }
                    }
                }
                for idx in df.index
            ]
            tasks_added = self.project.import_tasks(tasks=data_to_be_added)
            df_idx['task_id'] = tasks_added
            return df_idx

        def get_annotations_from_ls(
            ds: DataStore, input_dts: List[DataTable], output_dts: List[DataTable],
            run_config: RunConfig
        ):
            """
                Записывает в табличку задачи из сервера LS вместе с разметкой согласно
                дате последней синхронизации
            """
            # created_ago - очень плохой параметр, он меняется каждый раз, когда происходит запрос
            def _cleanup(values):
                for ann in values:
                    if 'created_ago' in ann:
                        del ann['created_ago']
                return values

            sync_datetime_df = sync_datetime_dt.get_data(idx=pd.DataFrame({'project_id': [self.project.id]}))
            if sync_datetime_df.empty:
                last_sync = datetime.fromtimestamp(0)
            else:
                last_sync = sync_datetime_df.loc[0, 'datetime']

            now = datetime.now()
            filters = Filters.create(
                conjunction="and", items=[
                    Filters.item(
                        name="tasks:updated_at",  # в sdk нету Column_LS.updated_at
                        operator=Operator.GREATER_OR_EQUAL,
                        column_type=Type.Datetime,
                        value=Filters.value(value=Filters.datetime(last_sync))
                    )
                ]
            )
            for tasks_page in get_tasks_iter(self.project, filters=filters):
                output_df = pd.DataFrame.from_records(
                    {
                        **{
                            primary_key: [task['data'][primary_key] for task in tasks_page]
                            for primary_key in self.primary_keys
                        },
                        'annotations': [_cleanup(task['annotations']) for task in tasks_page]
                    }
                )
                output_dts[0].store_chunk(output_df, now=now.timestamp())

            sync_datetime_df.loc[0, 'datetime'] = now
            sync_datetime_dt.store_chunk(sync_datetime_df, now=now.timestamp())

        return [
            BatchTransformStep(
                name='upload_data_to_ls',
                func=upload_tasks,
                input_dts=[input_dt],
                output_dts=[output_uploader_dt],
            ),
            DatatableTransformStepNoChangeList(
                name='get_annotations_from_ls',
                func=get_annotations_from_ls,
                input_dts=[],
                output_dts=[output_dt],
                check_for_changes=False
            )
        ]
