format-black:
	autoflake -r --in-place --remove-all-unused-imports datapipe_label_studio_lite/ tests/
	black --verbose --config black.toml datapipe_label_studio_lite/ tests/
