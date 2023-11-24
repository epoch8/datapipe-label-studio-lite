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

from datapipe.compute import Catalog, Pipeline, Table
from datapipe.step.batch_transform import BatchTransform
from datapipe.step.datatable_transform import DatatableTransformStep
from datapipe.step.batch_generate import BatchGenerate, do_batch_generate
from datapipe.datatable import DataStore
from datapipe.store.database import TableStoreDB
from datapipe.compute import build_compute, run_steps

from datapipe_label_studio_lite.create_projects_step import CreateLabelStudioProjects
from datapipe_label_studio_lite.upload_tasks_pipeline import LabelStudioUploadTasksToProjects
from datapipe_label_studio_lite.upload_predictions_pipeline import LabelStudioUploadPredictionsToProjects
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


def gen_ls_project_setting():
    yield pd.DataFrame(
        {
            "project_identifier": ["project_identifier0", "project_identifier1"],
            "project_label_config_at_create": [PROJECT_LABEL_CONFIG_TEST, PROJECT_LABEL_CONFIG_TEST],
            "project_description_at_create": ["", ""],
        }
    )


def gen_data_df():
    yield pd.DataFrame(
        {
            "project_identifier": [f"project_identifier0" for i in range(TASKS_COUNT)],
            "id": [f"task_{i}" for i in range(TASKS_COUNT)],
            "text": [np.random.choice([x for x in string.ascii_letters]) for i in range(TASKS_COUNT)],
        }
    )
    yield pd.DataFrame(
        {
            "project_identifier": [f"project_identifier1" for i in range(TASKS_COUNT)],
            "id": [f"task_{i}" for i in range(TASKS_COUNT)],
            "text": [np.random.choice([x for x in string.ascii_letters]) for i in range(TASKS_COUNT)],
        }
    )


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def convert_to_ls_input_data(data_df, include_preannotations: bool, include_prepredictions: bool):
    columns = ["project_identifier", "id", "text"]

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
    return data_df[["project_identifier", "id", "prediction"]]


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
        # project_title0 = f"[{request.node.callspec.id}/{workerid}]-0".replace("-", "")
        # project_title1 = f"[{request.node.callspec.id}/{workerid}]-1".replace("-", "")
        ds = DataStore(dbconn, create_meta_table=True)
        catalog = Catalog(
            {
                "ls_input_data_raw": Table(  # генерируем в тестах своим генератором
                    store=TableStoreDB(
                        dbconn=dbconn,
                        name="ls_input_data_raw",
                        data_sql_schema=[
                            Column("project_identifier", String(), primary_key=True),
                            Column("id", String(), primary_key=True),
                            Column("text", String()),
                        ],
                        create_table=True,
                    )
                ),
                "ls_project_setting": Table(
                    store=TableStoreDB(
                        dbconn=dbconn,
                        name="ls_project_setting",
                        data_sql_schema=[
                            Column("project_identifier", String(), primary_key=True),
                            Column("project_label_config_at_create", String),
                            Column("project_description_at_create", String),
                        ],
                        create_table=True,
                    )
                ),
                "ls_input_data": Table(
                    store=TableStoreDB(
                        dbconn=dbconn,
                        name="ls_input_data",
                        data_sql_schema=[
                            Column("project_identifier", String(), primary_key=True),
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
                            Column("project_identifier", String(), primary_key=True),
                            Column("id", String(), primary_key=True),
                            Column("prediction", JSON),
                        ],
                        create_table=True,
                    )
                ),
            }
        )
        main_steps = [
            BatchGenerate(
                func=gen_ls_project_setting,
                outputs=["ls_project_setting"],
            ),
            BatchTransform(
                func=wrapped_partial(
                    convert_to_ls_input_data,
                    include_preannotations=include_preannotations,
                    include_prepredictions=include_prepredictions,
                ),
                inputs=["ls_input_data_raw"],
                outputs=["ls_input_data"],
            ),
            CreateLabelStudioProjects(
                input__label_studio_project_setting="ls_project_setting",
                output__label_studio_project="ls_project",
                ls_url=ls_url,
                api_key=api_key,
                create_table=True,
            ),
            LabelStudioUploadTasksToProjects(
                input__item="ls_input_data",
                input__label_studio_project="ls_project",
                output__label_studio_project_task="ls_task",
                output__label_studio_project_annotation="ls_output",
                output__label_studio_sync_table="ls_sync_datetime",
                ls_url=ls_url,
                api_key=api_key,
                project_label_config_at_create=PROJECT_LABEL_CONFIG_TEST,
                primary_keys=["project_identifier", "id"],
                columns=["text"],
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
                LabelStudioUploadPredictionsToProjects(
                    input__item__has__prediction="ls_input_data__has__prediction",
                    input__label_studio_project="ls_project",
                    input__label_studio_project_task="ls_task",
                    output__label_studio_project_prediction="ls_prediction",
                    ls_url=ls_url,
                    api_key=api_key,
                    primary_keys=["project_identifier", "id"],
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
        for project_title in ["project_identifier0", "project_identifier1"]:
            project = get_project_by_title(label_studio_session, project_title)
            if project is not None:
                project.delete_project(project.id)

        yield (
            ds,
            catalog,
            steps,
            include_preannotations,
            include_prepredictions,
            include_predictions,
            label_studio_session,
            delete_unannotated_tasks_only_on_update,
        )

        for project_title in ["project_identifier0", "project_identifier1"]:
            project = get_project_by_title(label_studio_session, project_title)
            if project is not None:
                project.delete_project(project.id)


def ls_moderation_base_many_projects(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
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

    for project_identifier in ["project_identifier0", "project_identifier1"]:
        idx_project = pd.DataFrame([{"project_identifier": project_identifier}])
        assert len(ds.get_table("ls_task").get_data(idx_project)) == TASKS_COUNT
        assert len(ds.get_table("ls_output").get_data(idx_project)) == TASKS_COUNT

        # Проверяем проверку на заливку уже размеченных данных
        if include_preannotations:
            assert len(ds.get_table("ls_output").get_data(idx_project)) == TASKS_COUNT
            df_annotation = ds.get_table("ls_output").get_data(idx_project)
            for idx in df_annotation.index:
                assert len(df_annotation.loc[idx, "annotations"]) == 1
                assert df_annotation.loc[idx, "annotations"][0]["result"][0]["value"]["choices"][0] in [
                    "Class1",
                    "Class2",
                ]

        # Person annotation imitation & incremental processing
        project = get_project_by_title(label_studio_session, project_identifier)
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
            idxs_df["project_identifier"] = project_identifier
            df_annotation = ds.get_table("ls_output").get_data(idx=idxs_df)
            if include_predictions:
                df_prediction = ds.get_table("ls_prediction").get_data(idx=idxs_df)
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

        assert len(ds.get_table("ls_output").get_data(idx_project)) == 10


@parametrize_with_cases(
    "ds, catalog, steps, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_moderation_many_projects(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
    include_preannotations: bool,
    include_prepredictions: bool,
    include_predictions: bool,
    label_studio_session: label_studio_sdk.Client,
    delete_unannotated_tasks_only_on_update: bool,
):
    ls_moderation_base_many_projects(
        ds=ds,
        catalog=catalog,
        steps=steps,
        include_preannotations=include_preannotations,
        include_prepredictions=include_prepredictions,
        include_predictions=include_predictions,
        label_studio_session=label_studio_session,
        delete_unannotated_tasks_only_on_update=delete_unannotated_tasks_only_on_update,
    )


@parametrize_with_cases(
    "ds, catalog, steps, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_when_data_is_changed_many_projects(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
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
        df = df1.copy()
        df["project_identifier"] = "project_identifier0"
        yield df
        df["project_identifier"] = "project_identifier1"
        yield df

    def _gen2():
        df = df2.copy()
        df["project_identifier"] = "project_identifier0"
        yield df
        df["project_identifier"] = "project_identifier1"
        yield df

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

    for project_identifier in ["project_identifier0", "project_identifier1"]:
        idx_project = pd.DataFrame([{"project_identifier": project_identifier}])
        project = get_project_by_title(label_studio_session, project_identifier)
        assert project is not None

        tasks = project.get_tasks()
        assert len(tasks) == TASKS_COUNT

        assert len(ds.get_table("ls_task").get_data(idx_project)) == TASKS_COUNT
        df_ls = ds.get_table("ls_output").get_data(idx_project)
        assert len(df_ls) == TASKS_COUNT
        if include_predictions:
            df_prediction = ds.get_table("ls_prediction").get_data(idx_project)
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
    "ds, catalog, steps, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_when_task_is_missing_from_ls_many_projects(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
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
        df = df1.copy()
        df["project_identifier"] = "project_identifier0"
        yield df
        df["project_identifier"] = "project_identifier1"
        yield df

    def _gen2():
        df = df2.copy()
        df["project_identifier"] = "project_identifier0"
        yield df
        df["project_identifier"] = "project_identifier1"
        yield df

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

    for project_identifier in ["project_identifier0", "project_identifier1"]:
        idx_project = pd.DataFrame([{"project_identifier": project_identifier}])
        project = get_project_by_title(label_studio_session, project_identifier)
        assert project is not None

        tasks = project.get_tasks()
        assert len(tasks) == TASKS_COUNT

        assert len(ds.get_table("ls_task").get_data(idx_project)) == TASKS_COUNT
        df_ls = ds.get_table("ls_output").get_data(idx_project)
        assert len(df_ls) == TASKS_COUNT
        if include_predictions:
            df_prediction = ds.get_table("ls_prediction").get_data(idx_project)
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
    "ds, catalog, steps, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_when_some_data_is_deleted_many_projects(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
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
    data_df = pd.concat(list(gen_data_df()), ignore_index=True)
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

    for project_identifier in ["project_identifier0", "project_identifier1"]:
        idx_project = pd.DataFrame([{"project_identifier": project_identifier}])
        project = get_project_by_title(label_studio_session, project_identifier)
        assert project is not None

        tasks = project.get_tasks()
        assert len(tasks) == TASKS_COUNT - 5

        df_ls_task = ds.get_table("ls_task").get_data(idx_project)
        assert len(df_ls_task) == TASKS_COUNT - 5

        df_ls = ds.get_table("ls_output").get_data(idx_project)
        assert len(df_ls) == TASKS_COUNT - 5
        if include_predictions:
            df_prediction = ds.get_table("ls_prediction").get_data(idx_project)
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
    "ds, catalog, steps, include_preannotations, include_prepredictions, "
    "include_predictions, label_studio_session, delete_unannotated_tasks_only_on_update",
    cases=CasesLabelStudio,
)
def test_ls_moderate_then_delete_task_many_projects(
    ds: DataStore,
    catalog: Catalog,
    steps: List[DatatableTransformStep],
    include_preannotations: bool,
    include_prepredictions: bool,
    include_predictions: bool,
    label_studio_session: label_studio_sdk.Client,
    delete_unannotated_tasks_only_on_update: bool,
):
    ls_moderation_base_many_projects(
        ds=ds,
        catalog=catalog,
        steps=steps,
        include_preannotations=include_preannotations,
        include_prepredictions=include_prepredictions,
        include_predictions=include_predictions,
        label_studio_session=label_studio_session,
        delete_unannotated_tasks_only_on_update=delete_unannotated_tasks_only_on_update,
    )
    dt = ds.get_table("ls_input_data_raw")
    dt.delete_by_idx(dt.get_data(pd.DataFrame({"id": [f"task_{i}" for i in range(5)]})))
    run_steps(ds, steps)
    for project_identifier in ["project_identifier0", "project_identifier1"]:
        idx_project = pd.DataFrame([{"project_identifier": project_identifier}])
        project = get_project_by_title(label_studio_session, project_identifier)
        tasks_after = project.get_tasks()
        if delete_unannotated_tasks_only_on_update:
            assert len(ds.get_table("ls_output").get_data(idx_project)) == 10
            assert len(tasks_after) == 10
            assert len(ds.get_table("ls_task").get_data(idx_project)) == 10
            assert len(ds.get_table("ls_output").get_data(idx_project)) == 10
        else:
            assert len(ds.get_table("ls_output").get_data(idx_project)) == 5
            assert len(tasks_after) == 5
            assert len(ds.get_table("ls_task").get_data(idx_project)) == 5
            assert len(ds.get_table("ls_output").get_data(idx_project)) == 5
