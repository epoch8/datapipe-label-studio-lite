import requests
import label_studio_sdk
from urllib.parse import urljoin
from typing import List, Optional


def sign_up(ls_url: str, email: str, password: str):
    session = requests.Session()
    response = session.get(
        url=urljoin(ls_url, 'user/signup/')
    )
    response_signup = session.post(
        url=urljoin(ls_url, 'user/signup/'),
        data={
            'csrfmiddlewaretoken': response.cookies['csrftoken'],
            'email': email,
            'password': password
        }
    )
    if not response_signup.ok:
        raise ValueError('Signup failed.')
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
    ).json()['token']
    return api_key


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
