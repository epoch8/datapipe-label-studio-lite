import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, SupportsInt, cast
from urllib.parse import urljoin

import requests
from label_studio_sdk import LabelStudio
from label_studio_sdk.core.api_error import ApiError

from datapipe_label_studio_lite.types import ProjectDict, StorageDict, ImportTasksResponseDict

def sign_up(ls_url: str, email: str, password: str) -> Optional[str]:
    session = requests.Session()
    response_first = session.get(url=urljoin(ls_url, "user/signup/"))
    response_signup = session.post(
        url=urljoin(ls_url, "user/signup/"),
        data={
            "csrfmiddlewaretoken": response_first.cookies["csrftoken"],
            "email": email,
            "password": password,
        },
    )
    if not response_signup.ok:
        raise ValueError("Signup failed.")
    api_key = session.get(url=urljoin(ls_url, "api/current-user/token")).json()
    if "token" in api_key:
        return api_key["token"]
    return None


def login_and_get_token(ls_url: str, email: str, password: str) -> str:
    session = requests.Session()
    response = session.get(url=urljoin(ls_url, "user/login/"))
    session.post(
        url=urljoin(ls_url, "user/login/"),
        data={
            "csrfmiddlewaretoken": response.cookies["csrftoken"],
            "email": email,
            "password": password,
        },
    )
    api_key = session.get(url=urljoin(ls_url, "api/current-user/token")).json()
    if "token" in api_key:
        return api_key["token"]
    else:
        raise ValueError("Login failed.")


def _object_to_dict(obj: object) -> Dict[str, object]:
    if isinstance(obj, dict):
        return cast(Dict[str, object], obj)
    if hasattr(obj, "model_dump"):
        return cast(Dict[str, object], obj.model_dump())
    if hasattr(obj, "dict"):
        return cast(Dict[str, object], obj.dict())
    return cast(Dict[str, object], dict(obj.__dict__))


def project_to_dict(project: object) -> ProjectDict:
    data = _object_to_dict(project)
    project_id = data.get("id")
    if project_id is None:
        raise ValueError("Project response does not include 'id'.")
    title = data.get("title")
    if isinstance(project_id, (int, str)):
        data["id"] = int(project_id)
    else:
        data["id"] = int(cast(SupportsInt, project_id))
    data["title"] = title if isinstance(title, str) else str(title or "")
    return cast(ProjectDict, data)


def storage_to_dict(storage: object) -> StorageDict:
    data = _object_to_dict(storage)
    return cast(StorageDict, data)


def import_tasks_response_to_dict(response: object) -> ImportTasksResponseDict:
    data = _object_to_dict(response)
    return cast(ImportTasksResponseDict, data)


def get_project_by_title(ls: LabelStudio, title: str) -> Optional[ProjectDict]:
    candidates: List[ProjectDict] = []
    for page_items in _iter_paged_items(
        ls.projects.list,
        title=title,
        page_size=100,
    ):
        candidates.extend(project_to_dict(item) for item in page_items)

    titles = [project.get("title") for project in candidates]
    if title in titles:
        if titles.count(title) > 1:
            raise ValueError(f'There are 2 or more projects with title="{title}"')
        return candidates[titles.index(title)]
    return None


def _resolve_base_url(ls: LabelStudio) -> Optional[str]:
    for attr in ["base_url", "_base_url"]:
        value = getattr(ls, attr, None)
        if value:
            return value
    for attr in ["client", "_client"]:
        client = getattr(ls, attr, None)
        if client:
            value = getattr(client, "base_url", None) or getattr(client, "_base_url", None)
            if value:
                return value
    return None


def is_service_up(ls: LabelStudio, raise_exception: bool = False) -> bool:
    try:
        base_url = _resolve_base_url(ls)
        if not base_url:
            host = os.environ.get("LABEL_STUDIO_HOST", "localhost")
            port = os.environ.get("LABEL_STUDIO_PORT", "8080")
            base_url = f"http://{host}:{port}"
        requests.head(base_url)
        return True
    except requests.exceptions.ConnectionError:
        if raise_exception:
            raise
        else:
            return False


def _task_to_dict(task: Any) -> Dict[str, Any]:
    return _object_to_dict(task)


def _iter_paged_items(
    list_func,
    *,
    page_size: int,
    **kwargs,
) -> Iterator[List[Any]]:
    page = 1
    while True:
        try:
            response = list_func(page=page, page_size=page_size, **kwargs)
        except ApiError as exc:
            if getattr(exc, "status_code", None) == 404:
                break
            raise
        items = getattr(response, "items", None)
        if items is None:
            items = getattr(response, "results", None)
        if items is None and isinstance(response, dict):
            items = response.get("results", response.get("tasks", []))
        items = items or []
        if len(items) == 0:
            break
        yield items
        if len(items) < page_size:
            break
        page += 1


def get_tasks_iter(
    client: LabelStudio,
    project_id: int,
    filters: Optional[Dict[str, Any]] = None,
    ordering: Optional[Sequence[str]] = None,
    view_id: Optional[int] = None,
    selected_ids: Optional[Sequence[int]] = None,
    only_ids: bool = False,
    page_size: int = 100,
) -> Iterator[List[Dict[str, Any]]]:
    """Retrieve a subset of tasks from the Data Manager based on a filter, ordering mechanism, or a
    predefined view ID.

    Parameters
    ----------
    filters: label_studio_sdk.data_manager.Filters.create()
        JSON objects representing Data Manager filters. Use `label_studio_sdk.data_manager.Filters.create()`
        helper to create it.
        Example:
    ```json
    {
        "conjunction": "and",
        "items": [
        {
            "filter": "filter:tasks:id",
            "operator": "equal",
            "type": "Number",
            "value": 1
        }
        ]
    }
    ```
    ordering: list of label_studio_sdk.data_manager.Column
        List with <b>one</b> string representing Data Manager ordering.
        Use `label_studio_sdk.data_manager.Column` helper class.
        Example:
        ```[Column.total_annotations]```, ```['-' + Column.total_annotations]``` - inverted order
    view_id: int
        View ID, visible as a Data Manager tab, for which to retrieve filters, ordering, and selected items
    selected_ids: list of ints
        Task IDs
    only_ids: bool
        If true, return only task IDs

    Returns
    -------
    list
        Task list with task data, annotations, predictions and other fields from the Data Manager

    """

    query: Dict[str, Any] = {}
    if filters is not None:
        query["filters"] = filters
    if ordering is not None:
        query["ordering"] = list(ordering)
    if selected_ids is not None:
        query["selectedItems"] = {
            "all": False,
            "included": [int(task_id) for task_id in selected_ids],
        }

    for page_items in _iter_paged_items(
        client.tasks.list,
        project=project_id,
        query=json.dumps(query) if query else None,
        view=view_id,
        page_size=page_size,
        fields="task_only" if only_ids else "all",
    ):
        yield [_task_to_dict(task) for task in page_items]
