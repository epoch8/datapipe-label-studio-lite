# 0.3.1
* Added `Python 3.11` support

# 0.3.0
* Update datapipe-core version (0.13.0-alpha.4)
* `LabelStudioStep` now supports deleting tasks in `upload` table. In `output` table older annotations are not deleted.
* Added argument `delete_unannotated_tasks_only_on_update` in `LabelStudioStep`.
* Removed some dependencies

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