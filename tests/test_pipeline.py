from typing import List

import pytest
from pytest_cases import parametrize_with_cases, parametrize

import time
import string
from functools import partial, update_wrapper
import pandas as pd
import numpy as np

from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import String
from pkg_resources import parse_version

from datapipe.types import data_to_index
from datapipe.compute import Catalog, Pipeline, Table
from datapipe.step.batch_transform import BatchTransform
from datapipe.step.datatable_transform import DatatableTransformStep
from datapipe.step.batch_generate import do_batch_generate
from datapipe.datatable import DataStore
from datapipe.store.database import TableStoreDB
from datapipe.compute import build_compute, run_steps

from datapipe_label_studio_lite.pipeline import LabelStudioStep
import label_studio_sdk
from datapipe_label_studio_lite.sdk_utils import get_project_by_title, is_service_up


PROJECT_LABEL_CONFIG_TEST = """<View>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text" choice="single" showInLine="true">
    <Choice value="Class1"/>
    <Choice value="Class2"/>
    <Choice value="Class1_annotation"/>
    <Choice value="Class2_annotation"/>
  </Choices>
</View>"""


def wait_until_label_studio_is_up(ls: label_studio_sdk.Client):
    raise_exception = False
    counter = 0
    while not is_service_up(ls, raise_exception=raise_exception):
        time.sleep(1.0)
        counter += 1
        if counter >= 60:
            raise_exception = True


TASKS_COUNT = 10


def gen_data_df():
    yield pd.DataFrame(
        {
            "id": [f"task_{i}" for i in range(TASKS_COUNT)],
            "text": [
                np.random.choice([x for x in string.ascii_letters])
                for i in range(TASKS_COUNT)
            ],
        }
    )


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def convert_to_ls_input_data(
    data_df, include_preannotations: bool, include_prepredictions: bool
):
    columns = ["id", "text"]

    for column, bool_ in [
        ("preannotations", include_preannotations),
        ("prepredictions", include_prepredictions),
    ]:
        if bool_:
            data_df[column] = [
                [
                    {
                        "result": [
                            {
                                "value": {
                                    "choices": [np.random.choice(["Class1", "Class2"])]
                                },
                                "from_name": "label",
                                "to_name": "text",
                                "type": "choices",
                            }
                        ]
                    }
                ]
                for _ in range(len(data_df))
            ]
            columns.append(column)

    return data_df[columns]


def add_predictions(data_df):
    columns = ["id", "predictions", "model_version"]
    data_df["predictions"] = [
        {
            "result": [
                {
                    "value": {"choices": [np.random.choice(["Class1", "Class2"])]},
                    "from_name": "label",
                    "to_name": "text",
                    "type": "choices",
                }
            ]
        }
        for _ in range(len(data_df))
    ]
    data_df["model_version"] = "test-model"

    return data_df[columns]


INCLUDE_PARAMS = [
    pytest.param(
        {"include_preannotations": False, "include_prepredictions": False}, id=""
    ),
    # pytest.param(
    #     {
    #         'include_preannotations': True,
    #         'include_prepredictions': False
    #     },
    #     id='Preann'
    # ),
    # pytest.param(
    #     {
    #         'include_preannotations': False,
    #         'include_prepredictions': True
    #     },
    #     id='Prepred'
    # ),
    # pytest.param(
    #     {
    #         'include_preannotations': True,
    #         'include_prepredictions': True
    #     },
    #     id='PreannPrepred'
    # ),
]

INCLUDE_PREDICTIONS = [
    pytest.param(False, id="NoPredsStep"),
    # pytest.param(
    #     True,
    #     id='WithPredStep'
    # ),
]

DELETE_UNANNOTATED_TASKS_ONLY_ON_UPDATE = [
    pytest.param(False, id="DelAllOnUpdate"),
    pytest.param(True, id="DelUnAnnOnUpdate"),
    # pytest.param(
    #     True,
    #     id='WithPredStep'
    # ),
]


class CasesLabelStudio:
    @parametrize("include_predictions", INCLUDE_PREDICTIONS)
    @parametrize("include_params", INCLUDE_PARAMS)
    @parametrize(
        "delete_unannotated_tasks_only_on_update",
        DELETE_UNANNOTATED_TASKS_ONLY_ON_UPDATE,
    )
    def case_ls(
        self,
        include_params,
        include_predictions,
        dbconn,
        ls_url_and_api_key,
        request,
        delete_unannotated_tasks_only_on_update,
    ):
        if hasattr(request.config, "workerinput"):
            workerid = request.config.workerinput["workerid"]
        else:
            workerid = "master"
        ls_url, api_key = ls_url_and_api_key
        include_preannotations, include_prepredictions = (
            include_params["include_preannotations"],
            include_params["include_prepredictions"],
        )
        project_title = f"[{request.node.callspec.id}/{workerid}]".replace("-", "")
        ds = DataStore(dbconn, create_meta_table=True)
        catalog = Catalog(
            {
                "ls_input_data_raw": Table(  # генерируем в тестах своим генератором
                    store=TableStoreDB(
                        dbconn=dbconn,
                        name="ls_input_data_raw",
                        data_sql_schema=[
                            Column("id", String(), primary_key=True),
                            Column("text", String()),
                        ],
                        create_table=True,
                    )
                ),
                "ls_input_data": Table(
                    store=TableStoreDB(
                        dbconn=dbconn,
                        name="ls_input_data",
                        data_sql_schema=[
                            Column("id", String(), primary_key=True),
                            Column("text", String()),
                        ],
                        create_table=True,
                    )
                ),
            }
        )
        pipeline = Pipeline(
            [
                BatchTransform(
                    func=wrapped_partial(
                        convert_to_ls_input_data,
                        include_preannotations=include_preannotations,
                        include_prepredictions=include_prepredictions,
                    ),
                    inputs=["ls_input_data_raw"],
                    outputs=["ls_input_data"],
                ),
                LabelStudioStep(
                    input="ls_input_data",
                    output="ls_output",
                    sync_table="ls_sync_datetime",
                    ls_url=ls_url,
                    api_key=api_key,
                    dbconn=dbconn,
                    project_identifier=project_title,
                    project_label_config_at_create=PROJECT_LABEL_CONFIG_TEST,
                    data_sql_schema=[
                        Column("id", String(), primary_key=True),
                        Column("text", String()),
                    ],
                    create_table=True,
                    delete_unannotated_tasks_only_on_update=delete_unannotated_tasks_only_on_update,
                ),
            ]
        )
        # predictions_step = LabelStudioPredictionsStep(
        #     input='ls_input_data_raw',
        #     output='ls_predictions',
        #     ls_url=ls_url,
        #     api_key=api_key,
        #     project_identifier=project_title,
        #     primary_data_sql_schema=[
        #         Column('id', String(), primary_key=True),
        #     ],
        #     predictions_column='predictions',
        #     dbconn=dbconn
        # )
        # pipeline = Pipeline([main_step] + ([predictions_step] if include_predictions else []))

        steps = build_compute(ds, catalog, pipeline)
        label_studio_session = label_studio_sdk.Client(url=ls_url, api_key=api_key)
        wait_until_label_studio_is_up(label_studio_session)
        project = get_project_by_title(label_studio_session, project_title)
        if project is not None:
            project.delete_project(project.id)

        yield (
            ds,
            catalog,
            steps,
            project_title,
            include_preannotations,
            include_prepredictions,
            include_predictions,
            label_studio_session,
            delete_unannotated_tasks_only_on_update,
        )

        project = get_project_by_title(label_studio_session, project_title)
        if project is not None:
            project.delete_project(project.id)


@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_moderation(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
    project_title: str,
    include_preannotations: bool,
    include_prepredictions: bool,
    include_predictions: bool,
    label_studio_session: label_studio_sdk.Client,
    delete_unannotated_tasks_only_on_update: bool,
):
    # This should be ok (project will be created, but without data)
    run_steps(ds, steps)
    run_steps(ds, steps)

    # These steps should upload tasks
    do_batch_generate(
        func=gen_data_df,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    run_steps(ds, steps)
    assert len(ds.get_table("ls_output").get_data()) == TASKS_COUNT

    # Проверяем проверку на заливку уже размеченных данных
    if include_preannotations:
        assert len(ds.get_table("ls_output").get_data()) == TASKS_COUNT
        df_annotation = ds.get_table("ls_output").get_data()
        for idx in df_annotation.index:
            assert len(df_annotation.loc[idx, "annotations"]) == 1
            assert df_annotation.loc[idx, "annotations"][0]["result"][0]["value"][
                "choices"
            ][0] in ["Class1", "Class2"]

    # Person annotation imitation & incremental processing
    project = get_project_by_title(label_studio_session, project_title)
    assert project is not None

    tasks_res = project.get_tasks()
    assert len(tasks_res) == TASKS_COUNT

    run_steps(ds, steps)

    # Check that after second run no tasks are leaking
    tasks_res = project.get_tasks()
    assert len(tasks_res) == TASKS_COUNT

    tasks = np.array(tasks_res)
    for idxs in [[0, 3, 6, 7, 9], [1, 2, 4, 5, 8]]:
        annotations = [
            {
                "result": [
                    {
                        "value": {"choices": [np.random.choice(["Class1", "Class2"])]},
                        "from_name": "label",
                        "to_name": "text",
                        "type": "choices",
                    }
                ],
                "task": task["id"],
            }
            for task in tasks[idxs]
        ]
        for task, annotation in zip(tasks[idxs], annotations):
            label_studio_session.make_request(
                "POST",
                f"api/tasks/{task['id']}/annotations/",
                json=dict(
                    result=annotation["result"], was_cancelled=False, task_id=task["id"]
                ),
            )
        run_steps(ds, steps)
        idxs_df = pd.DataFrame.from_records(
            {"id": [task["data"]["id"] for task in tasks[idxs]]}
        )
        df_annotation = ds.get_table("ls_output").get_data(
            idx=data_to_index(idxs_df, ["id"])
        )
        for idx in df_annotation.index:
            assert len(df_annotation.loc[idx, "annotations"]) == (
                1 + include_preannotations
            )
            assert df_annotation.loc[idx, "annotations"][0]["result"][0]["value"][
                "choices"
            ][0] in (["Class1", "Class2"])
            # if include_predictions:
            #     assert len(df_annotation.loc[idx, 'predictions']) == include_prepredictions + include_predictions
            #     if include_prepredictions or include_predictions:
            #         assert df_annotation.loc[idx, 'predictions'][0]['result'][0]['value']['choices'][0] in (
            #             ["Class1", "Class2"]
            #         )


@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_when_data_is_changed(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
    project_title: str,
    include_preannotations: bool,
    include_prepredictions: bool,
    include_predictions: bool,
    label_studio_session: label_studio_sdk.Client,
    delete_unannotated_tasks_only_on_update: bool,
):
    df1 = pd.DataFrame(
        {
            "id": [f"task_{i}" for i in range(TASKS_COUNT)],
            "text": (
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
                + ["a"] * (TASKS_COUNT % 10)
            )
            * (TASKS_COUNT // 10),
        }
    )

    df2 = pd.DataFrame(
        {
            "id": [f"task_{i}" for i in range(TASKS_COUNT)],
            "text": (
                ["A", "B", "C", "d", "E", "f", "G", "h", "I", "j"]
                + ["a"] * (TASKS_COUNT % 10)
            )
            * (TASKS_COUNT // 10),
        }
    )

    def _gen():
        yield df1

    def _gen2():
        yield df2

    # Upload tasks
    do_batch_generate(
        func=_gen,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    run_steps(ds, steps)

    # These steps should delete old tasks and create new tasks
    do_batch_generate(
        func=_gen2,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    run_steps(ds, steps)

    project = get_project_by_title(label_studio_session, project_title)
    assert project is not None

    tasks = project.get_tasks()
    assert len(tasks) == TASKS_COUNT

    df_ls = ds.get_table("ls_output").get_data()

    for idx in df_ls.index:
        # Разметка не должна уйти:
        if include_preannotations and delete_unannotated_tasks_only_on_update:
            assert len(df_ls.loc[idx, "annotations"]) > 0
        # Предсказания не должны уйти
        # if include_predictions:
        #     assert len(df_ls.loc[idx, 'predictions']) == include_prepredictions + include_predictions


@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_when_task_is_missing_from_ls(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
    project_title: str,
    include_preannotations: bool,
    include_prepredictions: bool,
    include_predictions: bool,
    label_studio_session: label_studio_sdk.Client,
    delete_unannotated_tasks_only_on_update: bool,
):
    df1 = pd.DataFrame(
        {
            "id": [f"task_{i}" for i in range(TASKS_COUNT)],
            "text": (
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
                + ["a"] * (TASKS_COUNT % 10)
            )
            * (TASKS_COUNT // 10),
        }
    )

    df2 = pd.DataFrame(
        {
            "id": [f"task_{i}" for i in range(TASKS_COUNT)],
            "text": (
                ["A", "B", "C", "d", "E", "f", "G", "h", "I", "j"]
                + ["a"] * (TASKS_COUNT % 10)
            )
            * (TASKS_COUNT // 10),
        }
    )

    def _gen():
        yield df1

    def _gen2():
        yield df2

    # Upload tasks
    do_batch_generate(
        func=_gen,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    run_steps(ds, steps)

    upload_dt = catalog.get_datatable(ds, "ls_input_data_upload")
    upload_dt.store_chunk(
        pd.DataFrame(
            {
                "id": ["task_0"],
                "task_id": [-1],
            }
        )
    )

    # These steps should delete old tasks and create new tasks
    do_batch_generate(
        func=_gen2,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    run_steps(ds, steps)

    project = get_project_by_title(label_studio_session, project_title)
    assert project is not None

    tasks = project.get_tasks()
    assert len(tasks) == TASKS_COUNT + 1

    df_ls = ds.get_table("ls_output").get_data()

    for idx in df_ls.index:
        # Разметка не должна уйти:
        if include_preannotations and delete_unannotated_tasks_only_on_update:
            assert len(df_ls.loc[idx, "annotations"]) > 0
        # Предсказания не должны уйти
        # if include_predictions:
        #     assert len(df_ls.loc[idx, 'predictions']) == include_prepredictions + include_predictions


@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_when_some_data_is_deleted(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
    project_title: str,
    include_preannotations: bool,
    include_prepredictions: bool,
    include_predictions: bool,
    label_studio_session: label_studio_sdk.Client,
    delete_unannotated_tasks_only_on_update: bool,
):
    # Skip this test when LS is 1.4.0 and include_preannotations=True, include_prepredictions=False
    if (
        include_preannotations
        and not include_prepredictions
        and (parse_version(label_studio_session.version) == parse_version("1.4.0"))
    ):
        return
    # These steps should upload tasks
    data_df = next(gen_data_df())
    data_df2 = (
        data_df.set_index("id")
        .drop(index=[f"task_{i}" for i in [0, 3, 5, 7, 9]])
        .reset_index()
    )

    def _gen():
        yield data_df

    def _gen2():
        yield data_df2

    do_batch_generate(
        func=_gen,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    run_steps(ds, steps)

    # Change 5 input elements
    do_batch_generate(
        func=_gen2,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    # These steps should delete tasks with same id accordingly, as data input has changed
    run_steps(ds, steps)

    project = get_project_by_title(label_studio_session, project_title)
    assert project is not None

    tasks = project.get_tasks()
    assert len(tasks) == TASKS_COUNT - 5

    df_ls_upload = ds.get_table("ls_input_data_upload").get_data()
    assert len(df_ls_upload) == TASKS_COUNT - 5

    df_ls = ds.get_table("ls_output").get_data()
    assert len(df_ls) == TASKS_COUNT - 5

    for idx in df_ls.index:
        # Разметка не должна уйти:
        if include_preannotations and delete_unannotated_tasks_only_on_update:
            assert len(df_ls.loc[idx, "annotations"]) > 0
        # Предсказания не должны уйти
        # if include_predictions:
        #     assert len(df_ls.loc[idx, 'predictions']) == include_prepredictions + include_predictions


@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_specific_updating_scenary(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
    project_title: str,
    include_preannotations: bool,
    include_prepredictions: bool,
    include_predictions: bool,
    label_studio_session: label_studio_sdk.Client,
    delete_unannotated_tasks_only_on_update: bool,
):
    df1 = pd.DataFrame(
        {
            "id": [f"task_{i}" for i in range(10)],
            "text": ["a", "b", "c", "d", "e", "0", "1", "2", "3", "4"],
        }
    )

    df2 = pd.DataFrame(
        {
            "id": [f"task_{i}" for i in range(9)] + ["task_10"],
            "text": ["A", "B", "C", "d", "e", "0", "10", "20", "30", "111"],
        }
    )

    def _gen():
        yield df1

    def _gen2():
        yield df2

    do_batch_generate(
        func=_gen,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    run_steps(ds, steps)

    # Add 5 annotations
    project = get_project_by_title(label_studio_session, project_title)
    assert project is not None
    tasks_before = project.get_tasks()
    tasks_before_sorted = np.array(
        sorted(tasks_before, key=lambda task: task["data"]["id"])
    )
    assert len(tasks_before) == 10
    tasks_ids_before = [task["id"] for task in tasks_before]

    # Добавляем разметку к половине задач
    annotations = [
        {
            "result": [
                {
                    "value": {"choices": [np.random.choice(["Class1", "Class2"])]},
                    "from_name": "label",
                    "to_name": "text",
                    "type": "choices",
                }
            ],
            "task": task["id"],
        }
        for task in tasks_before_sorted[[0, 1, 2, 3, 4]]
    ]
    for task, annotation in zip(tasks_before, annotations):
        label_studio_session.make_request(
            "POST",
            f"api/tasks/{task['id']}/annotations/",
            json=dict(
                result=annotation["result"], was_cancelled=False, task_id=task["id"]
            ),
        )

    # Получаем текущую полученную разметку
    run_steps(ds, steps)

    # Change 6 input elements at [0, 1, 2, 6, 7, 8], delete 1 input at [9] and add 1 input at [10]
    do_batch_generate(
        func=_gen2,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    # Табличка с лейбел студией должна обновиться
    run_steps(ds, steps)

    tasks_after = project.get_tasks()
    assert len(tasks_after) == 10

    df_ls_upload = ds.get_table("ls_input_data_upload").get_data()
    assert len(df_ls_upload) == 10

    df_ls = ds.get_table("ls_output").get_data()
    assert len(df_ls) == 10

    df_ls = pd.merge(df_ls_upload, df_ls)
    for idx in df_ls.index:
        if df_ls.loc[idx, "id"] in [f"task_{i}" for i in [0, 1, 2, 6, 7, 8]]:
            # Разметка при обновлении задачи не должна уйти, если delete_unannotated_tasks_only_on_update=False
            if (
                df_ls.loc[idx, "id"] in [f"task_{i}" for i in [0, 1, 2]]
                and delete_unannotated_tasks_only_on_update
            ):
                assert len(df_ls.loc[idx, "annotations"]) > 0
                assert df_ls.loc[idx, "task_id"] in tasks_ids_before
            else:
                assert len(df_ls.loc[idx, "annotations"]) == 0
        else:
            # Айдишники оставшихся неизмененных задач не должны поменяться:
            if df_ls.loc[idx, "id"] in [f"task_{i}" for i in [3, 4]]:
                assert len(df_ls.loc[idx, "annotations"]) > 0
            else:
                assert len(df_ls.loc[idx, "annotations"]) == 0

            if df_ls.loc[idx, "id"] not in [f"task_{i}" for i in [10]]:
                assert df_ls.loc[idx, "task_id"] in tasks_ids_before

        # Предсказания не должны уйти
        # if include_predictions:
        #     assert len(df_ls.loc[idx, 'predictions']) == include_prepredictions + include_predictions


@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_moderation_with_duplicates_in_ls(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
    project_title: str,
    include_preannotations: bool,
    include_prepredictions: bool,
    include_predictions: bool,
    label_studio_session: label_studio_sdk.Client,
    delete_unannotated_tasks_only_on_update: bool,
):
    # This should be ok (project will be created, but without data)
    run_steps(ds, steps)
    run_steps(ds, steps)

    # Загружаем данные для задач в LS во входную таблицу.
    do_batch_generate(
        func=gen_data_df,
        ds=ds,
        output_dts=[ds.get_table("ls_input_data_raw")],
    )
    
    # Добавляем дубликаты задач напрямую в проект LS.
    tasks_duplicates_to_add = [
        {
            "data": {
                "id": "task_1",
                "text": "task_1_new_text"
            }
        },
        {
            "data": {
                "id": "task_2",
                "text": "task_2_new_text"
            }
        }
    ]
    project = get_project_by_title(label_studio_session, project_title)
    project.import_tasks(tasks=tasks_duplicates_to_add)

    # Запускаем трансформацию.
    run_steps(ds, steps)
    
    # Проверяем количество задач в LS и данных в выходной таблице трубы.
    assert len(project.get_tasks()) == TASKS_COUNT + len(tasks_duplicates_to_add)
    assert len(ds.get_table("ls_output").get_data()) == TASKS_COUNT
