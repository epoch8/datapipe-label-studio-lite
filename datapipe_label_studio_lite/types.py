import logging
from typing import Union, List, Optional
from dataclasses import dataclass


logger = logging.getLogger("dataipipe_label_studio_lite")


@dataclass
class GCSBucket:
    bucket: str
    google_application_credentials: Optional[str] = None

    @property
    def type(self):
        return "gcs"


@dataclass
class S3Bucket:
    bucket: str
    key: str
    secret: str
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None

    @property
    def type(self):
        return "s3"


@dataclass
class Buckets:
    buckets: List[Union[GCSBucket, S3Bucket]]
