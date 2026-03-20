black:
	autoflake -r --in-place --remove-all-unused-imports datapipe_label_studio_lite/ tests/
	isort --profile black toml datapipe_label_studio_lite/ tests/
	black --verbose --config black.toml datapipe_label_studio_lite/ tests/

mypy:
	mypy -p datapipe_label_studio_lite --ignore-missing-imports --follow-imports=silent --check-untyped-defs