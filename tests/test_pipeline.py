from typing import List

import pytest
from pytest_cases import parametrize_with_cases, parametrize

import time
import string
from functools import partial, update_wrapper
import pandas as pd
import numpy as np

from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import String, JSON
from pkg_resources import parse_version

from datapipe.types import data_to_index
from datapipe.compute import Catalog, Pipeline, Table
from datapipe.step.batch_transform import BatchTransform
from datapipe.step.datatable_transform import DatatableTransformStep
from datapipe.step.batch_generate import do_batch_generate
from datapipe.datatable import DataStore
from datapipe.store.database import TableStoreDB
from datapipe.compute import build_compute, run_steps

from datapipe_label_studio_lite.upload_tasks_pipeline import LabelStudioUploadTasks
from datapipe_label_studio_lite.upload_predictions_pipeline import LabelStudioUploadPredictions
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
            "text": [np.random.choice([x for x in string.ascii_letters]) for i in range(TASKS_COUNT)],
        }
    )


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def convert_to_ls_input_data(data_df, include_preannotations: bool, include_prepredictions: bool):
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
                                "value": {"choices": [np.random.choice(["Class1", "Class2"])]},
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
    data_df["prediction"] = [
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
    return data_df[["id", "prediction"]]


INCLUDE_PARAMS = [
    pytest.param({"include_preannotations": False, "include_prepredictions": False}, id=""),
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
    pytest.param(True, id="WithPredStep"),
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
                "ls_input_data__has__prediction": Table(
                    store=TableStoreDB(
                        dbconn=dbconn,
                        name="ls_input_data__has__prediction",
                        data_sql_schema=[
                            Column("id", String(), primary_key=True),
                            Column("prediction", JSON),
                        ],
                        create_table=True,
                    )
                ),
            }
        )
        main_steps = [
            BatchTransform(
                func=wrapped_partial(
                    convert_to_ls_input_data,
                    include_preannotations=include_preannotations,
                    include_prepredictions=include_prepredictions,
                ),
                inputs=["ls_input_data_raw"],
                outputs=["ls_input_data"],
            ),
            LabelStudioUploadTasks(
                input__item="ls_input_data",
                output__label_studio_project_task="ls_task",
                output__label_studio_project_annotation="ls_output",
                output__label_studio_sync_table="ls_sync_datetime",
                ls_url=ls_url,
                api_key=api_key,
                project_identifier=project_title,
                project_label_config_at_create=PROJECT_LABEL_CONFIG_TEST,
                columns=["id", "text"],
                create_table=True,
                delete_unannotated_tasks_only_on_update=delete_unannotated_tasks_only_on_update,
            ),
        ]
        predictions_steps = (
            [
                BatchTransform(
                    func=add_predictions,
                    inputs=["ls_input_data_raw"],
                    outputs=["ls_input_data__has__prediction"],
                ),
                LabelStudioUploadPredictions(
                    input__item__has__prediction="ls_input_data__has__prediction",
                    input__label_studio_project_task="ls_task",
                    output__label_studio_project_prediction="ls_prediction",
                    ls_url=ls_url,
                    api_key=api_key,
                    project_identifier=project_title,
                    columns=["id"],
                    create_table=True,
                ),
            ]
            if include_predictions
            else []
        )
        pipeline = Pipeline(main_steps + predictions_steps)

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


def ls_moderation_base(
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
    assert len(ds.get_table("ls_task").get_data()) == TASKS_COUNT
    assert len(ds.get_table("ls_output").get_data()) == TASKS_COUNT

    # Проверяем проверку на заливку уже размеченных данных
    if include_preannotations:
        assert len(ds.get_table("ls_output").get_data()) == TASKS_COUNT
        df_annotation = ds.get_table("ls_output").get_data()
        for idx in df_annotation.index:
            assert len(df_annotation.loc[idx, "annotations"]) == 1
            assert df_annotation.loc[idx, "annotations"][0]["result"][0]["value"]["choices"][0] in ["Class1", "Class2"]

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
                json=dict(result=annotation["result"], was_cancelled=False, task_id=task["id"]),
            )
        run_steps(ds, steps)
        idxs_df = pd.DataFrame.from_records({"id": [task["data"]["id"] for task in tasks[idxs]]})
        df_annotation = ds.get_table("ls_output").get_data(idx=data_to_index(idxs_df, ["id"]))
        if include_predictions:
            df_prediction = ds.get_table("ls_prediction").get_data(idx=data_to_index(idxs_df, ["id"]))
            assert len(df_prediction) == len(df_annotation)
            df_prediction = pd.merge(df_annotation, df_prediction)
        for idx in df_annotation.index:
            assert len(df_annotation.loc[idx, "annotations"]) == (1 + include_preannotations)
            assert df_annotation.loc[idx, "annotations"][0]["result"][0]["value"]["choices"][0] in (
                ["Class1", "Class2"]
            )
            # if include_predictions:
            #     assert len(df_annotation.loc[idx, 'predictions']) == include_prepredictions + include_predictions
            #     if include_prepredictions or include_predictions:
            #         assert df_annotation.loc[idx, 'predictions'][0]['result'][0]['value']['choices'][0] in (
            #             ["Class1", "Class2"]
            #         )

    assert len(ds.get_table("ls_output").get_data()) == 10


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
    ls_moderation_base(
        ds=ds,
        catalog=catalog,
        steps=steps,
        project_title=project_title,
        include_preannotations=include_preannotations,
        include_prepredictions=include_prepredictions,
        include_predictions=include_predictions,
        label_studio_session=label_studio_session,
        delete_unannotated_tasks_only_on_update=delete_unannotated_tasks_only_on_update,
    )


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
            "text": (["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] + ["a"] * (TASKS_COUNT % 10))
            * (TASKS_COUNT // 10),
        }
    )

    df2 = pd.DataFrame(
        {
            "id": [f"task_{i}" for i in range(TASKS_COUNT)],
            "text": (["A", "B", "C", "d", "E", "f", "G", "h", "I", "j"] + ["a"] * (TASKS_COUNT % 10))
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

    assert len(ds.get_table("ls_task").get_data()) == TASKS_COUNT
    df_ls = ds.get_table("ls_output").get_data()
    assert len(df_ls) == TASKS_COUNT
    if include_predictions:
        df_prediction = ds.get_table("ls_prediction").get_data()
        assert len(df_prediction) == len(df_ls)
        df_prediction = pd.merge(df_ls, df_prediction)

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
            "text": (["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] + ["a"] * (TASKS_COUNT % 10))
            * (TASKS_COUNT // 10),
        }
    )

    df2 = pd.DataFrame(
        {
            "id": [f"task_{i}" for i in range(TASKS_COUNT)],
            "text": (["A", "B", "C", "d", "E", "f", "G", "h", "I", "j"] + ["a"] * (TASKS_COUNT % 10))
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

    assert len(ds.get_table("ls_task").get_data()) == TASKS_COUNT
    df_ls = ds.get_table("ls_output").get_data()
    assert len(df_ls) == TASKS_COUNT
    if include_predictions:
        df_prediction = ds.get_table("ls_prediction").get_data()
        assert len(df_prediction) == len(df_ls)
        df_prediction = pd.merge(df_ls, df_prediction)

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
    data_df2 = data_df.set_index("id").drop(index=[f"task_{i}" for i in [0, 3, 5, 7, 9]]).reset_index()

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

    df_ls_task = ds.get_table("ls_task").get_data()
    assert len(df_ls_task) == TASKS_COUNT - 5

    df_ls = ds.get_table("ls_output").get_data()
    assert len(df_ls) == TASKS_COUNT - 5
    if include_predictions:
        df_prediction = ds.get_table("ls_prediction").get_data()
        assert len(df_prediction) == TASKS_COUNT - 5
        df_prediction = pd.merge(df_ls, df_prediction)

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
    tasks_before_sorted = np.array(sorted(tasks_before, key=lambda task: task["data"]["id"]))
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
            json=dict(result=annotation["result"], was_cancelled=False, task_id=task["id"]),
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

    df_ls_task = ds.get_table("ls_task").get_data()
    assert len(df_ls_task) == 10

    df_ls = ds.get_table("ls_output").get_data()
    assert len(df_ls) == 10
    if include_predictions:
        df_prediction = ds.get_table("ls_prediction").get_data()
        assert len(df_prediction) == 10
        df_prediction = pd.merge(df_ls, df_prediction)

    df_ls = pd.merge(df_ls_task, df_ls)
    for idx in df_ls.index:
        if df_ls.loc[idx, "id"] in [f"task_{i}" for i in [0, 1, 2, 6, 7, 8]]:
            # Разметка при обновлении задачи не должна уйти, если delete_unannotated_tasks_only_on_update=False
            if df_ls.loc[idx, "id"] in [f"task_{i}" for i in [0, 1, 2]] and delete_unannotated_tasks_only_on_update:
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
def test_ls_moderate_then_delete_task(
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
    ls_moderation_base(
        ds=ds,
        catalog=catalog,
        steps=steps,
        project_title=project_title,
        include_preannotations=include_preannotations,
        include_prepredictions=include_prepredictions,
        include_predictions=include_predictions,
        label_studio_session=label_studio_session,
        delete_unannotated_tasks_only_on_update=delete_unannotated_tasks_only_on_update,
    )
    ds.get_table("ls_input_data_raw").delete_by_idx(idx=pd.DataFrame({"id": [f"task_{i}" for i in range(5)]}))
    run_steps(ds, steps)
    project = get_project_by_title(label_studio_session, project_title)
    tasks_after = project.get_tasks()
    if delete_unannotated_tasks_only_on_update:
        assert len(ds.get_table("ls_output").get_data()) == 10
        assert len(tasks_after) == 10
        assert len(ds.get_table("ls_task").get_data()) == 10
        assert len(ds.get_table("ls_output").get_data()) == 10
    else:
        assert len(ds.get_table("ls_output").get_data()) == 5
        assert len(tasks_after) == 5
        assert len(ds.get_table("ls_task").get_data()) == 5
        assert len(ds.get_table("ls_output").get_data()) == 5
