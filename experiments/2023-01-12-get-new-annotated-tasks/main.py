from datetime import datetime
from typing import List

import json

from label_studio_sdk import LabelStudio
from label_studio_sdk.data_manager import Column, Filters, Operator, Type


def get_new_annotated_tasks(ls_url: str, ls_token: str, ls_project_id: str) -> List[int]:
    client = LabelStudio(base_url=ls_url, api_key=ls_token)
    project_filters = Filters.create(
        conjunction="and",
        items=[
            Filters.item(
                name=Column.total_annotations,
                operator=Operator.GREATER,
                column_type=Type.Number,
                value=Filters.value(value=0),
            ),
            Filters.item(
                name=Column.created_at,
                operator=Operator.GREATER_OR_EQUAL,
                column_type=Type.Datetime,
                value=Filters.value(value=Filters.datetime(datetime.now())),
            ),
        ],
    )
    query = json.dumps({"filters": project_filters})
    response = client.tasks.list(
        project=int(ls_project_id),
        query=query,
        fields="task_only",
        page=1,
        page_size=200,
    )
    tasks = getattr(response, "tasks", None)
    if tasks is None and isinstance(response, dict):
        tasks = response.get("tasks", [])
    return [task["id"] for task in tasks]
