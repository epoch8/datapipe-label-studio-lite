import numpy as np
import pandas as pd
from typing import Any, Callable, Union, List, Optional
from dataclasses import dataclass
from datapipe.run_config import RunConfig
from datapipe.store.database import TableStoreDB

from sqlalchemy import Integer, Column, JSON

from datapipe.types import ChangeList, DataDF
from datapipe.compute import PipelineStep, DataStore, Catalog, DatatableTransformStep
from datapipe.core_steps import BatchTransformStep
from datapipe.store.database import DBConn
import label_studio_sdk
from datapipe_label_studio_lite.sdk_utils import get_project_by_title
from datapipe.types import (
    DataDF, data_to_index, index_to_data,
)


class DatatableTransformStepNoChangeList(DatatableTransformStep):
    def run_changelist(self, ds: DataStore, changelist: ChangeList, run_config: RunConfig = None) -> ChangeList:
        return ChangeList()


class DatatableTransformStepWithChangeList(DatatableTransformStep):
    def run_changelist(self, ds: DataStore, changelist: ChangeList, run_config: RunConfig = None) -> ChangeList:
        self.func(ds, self.input_dts, self.output_dts, run_config)
        return changelist


@dataclass
class LabelStudioStep(PipelineStep):
    input: str
    output: str

    ls_url: str
    api_key: str
    dbconn: Union[DBConn, str]
    project_identifier: Union[str, int]  # project_title or id
    data_sql_schema: List[Column]

    project_label_config_at_create: str = ''
    project_description_at_create: str = ''

    input_convert_func: Callable[[DataDF], DataDF] = lambda df: df
    output_convert_func: Callable[[DataDF], DataDF] = lambda df: df

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
        self.label_studio_session = label_studio_sdk.Client(
            url=self.ls_url,
            api_key=self.api_key if isinstance(self.api_key, str) else None,
            credentials=self.api_key if isinstance(self.api_key, tuple) else None
        )

        self.project: Optional[label_studio_sdk.Project] = None

    def get_or_create_project(self, raise_exception: bool = False) -> label_studio_sdk.Project:
        if self.project is not None:
            return self.project

        if not self.label_studio_session.check_connection():
            return '-1'

        self.project = (
            self.project_identifier if self.project_identifier.isnumeric()
            else get_project_by_title(self.label_studio_session, self.project_identifier)
        )
        if self.project is None:
            if raise_exception:
                raise ValueError(f"Project with {self.project_identifier=} not found.")
            else:
                return '-1'

        if self.project is None:
            self.project = self.label_studio_session.start_project(
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

        return self.project

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
                data_sql_schema=self.primary_keys + [Column('task_id', Integer)],
                create_table=self.create_table
            )
        )
        output_dt = ds.get_or_create_table(
            self.output, TableStoreDB(
                dbconn=self.dbconn,
                name=self.output,
                data_sql_schema=self.primary_keys + [Column('task_id', Integer), Column('annotations', JSON)],
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

            df = self.input_convert_func(df)

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
                for task_id in df_to_be_deleted.loc[df_idx, 'task_id']:
                    self.get_or_create_project().make_request(
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
            tasks_added = self.get_or_create_project().import_tasks(tasks=data_to_be_added)
            df_idx['task_id'] = tasks_added
            return df_idx

        def get_annotations_from_ls(ds, input_dts, output_dts, run_config):
            """
                Возвращает все задачи из сервера LS вместе с разметкой
            """
            # created_ago - очень плохой параметр, он меняется каждый раз, когда происходит запрос
            def _cleanup(values):
                for ann in values:
                    if 'created_ago' in ann:
                        del ann['created_ago']
                return values

            tasks = self.get_or_create_project().get_tasks()  # TO BE ADDED: фильтрация
            tasks_split = np.array_split(tasks, max(1, len(tasks) // 1000))
            for tasks_page in tasks_split:
                output_df = pd.DataFrame.from_records(
                    {
                        **{
                            primary_key: [task['data'][primary_key] for task in tasks_page]
                            for primary_key in self.primary_keys
                        },
                        'task_id': [int(task['id']) for task in tasks_page],
                        'annotations': [_cleanup(task['annotations']) for task in tasks_page]
                    }
                )
                output_dt.store_chunk(output_df)

        return [
            BatchTransformStep(
                name=f'{self.output}_upload_data_to_ls',
                func=upload_tasks,
                input_dts=[input_dt],
                output_dts=[output_uploader_dt],
            ),
            DatatableTransformStepNoChangeList(
                name=f'{self.output}_dump_from_ls_to_db',
                func=get_annotations_from_ls,
                input_dts=[],
                output_dts=[output_dt],
                check_for_changes=False
            )
        ]
