from datetime import datetime
from typing import List

from label_studio_sdk import Client, Project
from label_studio_sdk.data_manager import Filters, Column, Operator, Type


def get_new_annotated_tasks(ls_url: str, ls_token: str, ls_project_id: str) -> List[int]:
    ls = Client(url=ls_url, api_key=ls_token)
    project = Project.get_from_id(client=ls, project_id=ls_project_id)
    project_filters = Filters.create(conjunction="and", items=[
        Filters.item(name=Column.total_annotations, operator=Operator.GREATER, column_type=Type.Number, value=Filters.value(value=0)),
        Filters.item(name=Column.created_at, operator=Operator.GREATER_OR_EQUAL, column_type=Type.Datetime, value=Filters.value(value=Filters.datetime(datetime.now())))
    ])
    return project.get_tasks(filters=project_filters, only_ids=True)
