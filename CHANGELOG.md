# 0.2.3

* Fix handling of duplicate entities at sync
* Add logging to `get_annotations_from_ls`
* Add labels `func` and `group` to steps

# 0.2.2

* Update datapipe-core version (0.12.0)

# 0.2.1

* Removed `sqlite3` from required dependencies in pyproject.toml

# 0.2.0

* Add `name` parameter to `LabelStudioStep`
* Add name prefixes to transformation steps
* Add workaround for `500` from LS when trying to delete non-existent task
* Remove cyclic dependency on input_uploader_dt for upload task

# 0.1.0

* First version