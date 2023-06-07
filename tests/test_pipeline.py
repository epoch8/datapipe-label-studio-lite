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
from datapipe.core_steps import do_batch_generate, BatchTransform, DatatableTransformStep
from datapipe.datatable import DataStore
from datapipe.store.database import TableStoreDB
from datapipe.compute import build_compute, run_steps
from tests.conftest import assert_df_equal

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


class CasesLabelStudio:
    @parametrize("include_predictions", INCLUDE_PREDICTIONS)
    @parametrize("include_params", INCLUDE_PARAMS)
    def case_ls(
        self, include_params, include_predictions, dbconn, ls_url_and_api_key, request
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
                "raw_data": Table(  # генерируем в тестах своим генератором
                    store=TableStoreDB(
                        dbconn=dbconn,
                        name="raw_data",
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
                    inputs=["raw_data"],
                    outputs=["ls_input_data"],
                ),
                LabelStudioStep(
                    input="ls_input_data",
                    output="ls_annotations",
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
                ),
            ]
        )
        # predictions_step = LabelStudioPredictionsStep(
        #     input='raw_data',
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
        )

        project = get_project_by_title(label_studio_session, project_title)
        if project is not None:
            project.delete_project(project.id)


@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session",
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
):
    # This should be ok (project will be created, but without data)
    run_steps(ds, steps)
    run_steps(ds, steps)

    # These steps should upload tasks
    do_batch_generate(
        func=gen_data_df,
        ds=ds,
        output_dts=[ds.get_table("raw_data")],
    )
    run_steps(ds, steps)
    assert len(ds.get_table("ls_annotations").get_data()) == TASKS_COUNT

    # Проверяем проверку на заливку уже размеченных данных
    if include_preannotations:
        assert len(ds.get_table("ls_annotations").get_data()) == TASKS_COUNT
        df_annotation = ds.get_table("ls_annotations").get_data()
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
        df_annotation = ds.get_table("ls_annotations").get_data(
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
    "include_predictions, label_studio_session",
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
        output_dts=[ds.get_table("raw_data")],
    )
    run_steps(ds, steps)

    # These steps should delete old tasks and create new tasks with same ids
    do_batch_generate(
        func=_gen2,
        ds=ds,
        output_dts=[ds.get_table("raw_data")],
    )
    run_steps(ds, steps)

    project = get_project_by_title(label_studio_session, project_title)
    assert project is not None

    tasks = project.get_tasks()
    assert len(tasks) == TASKS_COUNT

    df_ls = ds.get_table("ls_annotations").get_data()

    for idx in df_ls.index:
        # Разметка не должна уйти:
        if include_preannotations:
            assert len(df_ls.loc[idx, "annotations"]) > 0
        # Предсказания не должны уйти
        # if include_predictions:
        #     assert len(df_ls.loc[idx, 'predictions']) == include_prepredictions + include_predictions


@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session",
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
        output_dts=[ds.get_table("raw_data")],
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

    # These steps should delete old tasks and create new tasks with same ids
    do_batch_generate(
        func=_gen2,
        ds=ds,
        output_dts=[ds.get_table("raw_data")],
    )
    run_steps(ds, steps)

    project = get_project_by_title(label_studio_session, project_title)
    assert project is not None

    tasks = project.get_tasks()
    assert len(tasks) == TASKS_COUNT + 1

    df_ls = ds.get_table("ls_annotations").get_data()

    for idx in df_ls.index:
        # Разметка не должна уйти:
        if include_preannotations:
            assert len(df_ls.loc[idx, "annotations"]) > 0
        # Предсказания не должны уйти
        # if include_predictions:
        #     assert len(df_ls.loc[idx, 'predictions']) == include_prepredictions + include_predictions


@pytest.mark.skip(reason="LabelStudioStep doesn't support deleting yet")
@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session",
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
        output_dts=[ds.get_table("raw_data")],
    )
    run_steps(ds, steps)

    # Change 5 input elements
    do_batch_generate(
        func=_gen2,
        ds=ds,
        output_dts=[ds.get_table("raw_data")],
    )
    # These steps should delete tasks with same id accordingly, as data input has changed
    run_steps(ds, steps)

    project = get_project_by_title(label_studio_session, project_title)
    assert project is not None

    tasks = project.get_tasks()
    assert len(tasks) == TASKS_COUNT - 5

    df_ls = ds.get_table("ls_annotations").get_data()

    for idx in df_ls.index:
        # Разметка не должна уйти:
        if include_preannotations:
            assert len(df_ls.loc[idx, "annotations"]) > 0
        # Предсказания не должны уйти
        # if include_predictions:
        #     assert len(df_ls.loc[idx, 'predictions']) == include_prepredictions + include_predictions


@parametrize_with_cases(
    "ds, catalog, steps, project_title, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session",
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
):
    df1 = pd.DataFrame(
        {"id": [f"task_{i}" for i in range(5)], "text": ["a", "b", "c", "d", "e"]}
    )

    df2 = pd.DataFrame(
        {"id": [f"task_{i}" for i in range(5)], "text": ["A", "B", "C", "d", "e"]}
    )

    def _gen():
        yield df1

    def _gen2():
        yield df2

    do_batch_generate(
        func=_gen,
        ds=ds,
        output_dts=[ds.get_table("raw_data")],
    )
    run_steps(ds, steps)

    # Add 5 annotations
    project = get_project_by_title(label_studio_session, project_title)
    assert project is not None
    tasks_res = project.get_tasks()
    tasks_ndarr = np.array(tasks_res)

    # Добавляем разметку ко всем задачам
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
        for task in tasks_ndarr
    ]
    for task, annotation in zip(tasks_ndarr, annotations):
        label_studio_session.make_request(
            "POST",
            f"api/tasks/{task['id']}/annotations/",
            json=dict(
                result=annotation["result"], was_cancelled=False, task_id=task["id"]
            ),
        )

    # Change 3 input elements
    do_batch_generate(
        func=_gen2,
        ds=ds,
        output_dts=[ds.get_table("raw_data")],
    )
    # Табличка с лейбел студией должна обновиться
    run_steps(ds, steps)

    tasks_ls = project.get_tasks()
    assert len(tasks_ls) == 5

    df_ls = ds.get_table("ls_annotations").get_data()
    for idx in df_ls.index:
        if df_ls.loc[idx, "id"] in [f"task_{i}" for i in range(3)]:
            # Разметка при обновлении задачи должна уйти:
            assert len(df_ls.loc[idx, "annotations"]) == 0
        else:
            # Разметка оставшихся задач должны остаться:
            assert len(df_ls.loc[idx, "annotations"]) > 0

        # Предсказания не должны уйти
        # if include_predictions:
        #     assert len(df_ls.loc[idx, 'predictions']) == include_prepredictions + include_predictions
