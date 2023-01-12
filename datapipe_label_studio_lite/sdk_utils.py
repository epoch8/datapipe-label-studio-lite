import requests
import label_studio_sdk
from urllib.parse import urljoin
from typing import Any, Dict, Iterator, List, Optional


def sign_up(ls_url: str, email: str, password: str) -> Optional[str]:
    session = requests.Session()
    response_first = session.get(
        url=urljoin(ls_url, 'user/signup/')
    )
    response_signup = session.post(
        url=urljoin(ls_url, 'user/signup/'),
        data={
            'csrfmiddlewaretoken': response_first.cookies['csrftoken'],
            'email': email,
            'password': password
        }
    )
    if not response_signup.ok:
        raise ValueError('Signup failed.')
    api_key = session.get(
        url=urljoin(ls_url, 'api/current-user/token')
    ).json()
    if 'token' in api_key:
        return api_key['token']
    return None


def login_and_get_token(ls_url: str, email: str, password: str) -> str:
    session = requests.Session()
    response = session.get(
        url=urljoin(ls_url, 'user/login/')
    )
    session.post(
        url=urljoin(ls_url, 'user/login/'),
        data={
            'csrfmiddlewaretoken': response.cookies['csrftoken'],
            'email': email,
            'password': password
        }
    )
    api_key = session.get(
        url=urljoin(ls_url, 'api/current-user/token')
    ).json()
    if 'token' in api_key:
        return api_key['token']
    else:
        raise ValueError("Login failed.")


def get_project_by_title(ls: label_studio_sdk.Client, title: str) -> Optional[label_studio_sdk.Project]:
    projects: List[label_studio_sdk.Project] = ls.get_projects()
    titles = [project.get_params()['title'] for project in projects]
    if title in titles:
        assert titles.count(title) == 1, f'There are 2 or more projects with title="{title}"'
        return projects[titles.index(title)]
    return None


def is_service_up(ls: label_studio_sdk.Client, raise_exception: bool = False) -> bool:
    try:
        ls.session.head(ls.url)
        return True
    except requests.exceptions.ConnectionError:
        if raise_exception:
            raise
        else:
            return False


def get_tasks_iter(
    project: label_studio_sdk.Project,
    filters=None,
    ordering=None,
    view_id=None,
    selected_ids=None,
    only_ids: bool = False,
) -> Iterator[List[Dict[str, Any]]]:
    """ Retrieve a subset of tasks from the Data Manager based on a filter, ordering mechanism, or a
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

    page = 1
    while True:
        try:
            data = project.get_paginated_tasks(
                filters=filters,
                ordering=ordering,
                view_id=view_id,
                selected_ids=selected_ids,
                only_ids=only_ids,
                page=page,
                page_size=100
            )
            yield data['tasks']
            page += 1
        # we'll get 404 from API on empty page
        except label_studio_sdk.project.LabelStudioException:
            break
