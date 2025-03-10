from __future__ import annotations
from typing import List, Optional, Union, Any
import os
import traceback
from pathlib import Path

from google.cloud.storage import Client, Bucket, Blob
from google.api_core.page_iterator import Iterator
from google.cloud.exceptions import NotFound as GoogleCloudNotFound

from src.creds_accessor.CredentialAccessorAttr import CredentialAccessorAttribute as CrAttr


class GoogleCloudStorage:

    def __init__(
            self,
            attr: CrAttr,
            bucket_name: str
    ) -> None:
        """
        For creating, inserting, updating, and deleting content in GCS

        :param attr: credential attribute
        :param bucket_name:

       Info:
        . Last edited by: I PUTU MAHENDRA WIJAYA <i.wijaya@agriaku.com>
        . Last edited at: November 09, 2023
        . Copyright: DATA SCIENCE team, PT. Agriaku Digital Indonesia
        """

        super(GoogleCloudStorage, self).__init__()

        self.attr: CrAttr = attr
        self.client: Client = self.gcs_client()
        self.bucket_name: str = bucket_name
        self.bucket_ref: Bucket = self.get_bucket()
        self.folders: List[str] = []
        self.folder_ref: Optional[Blob] = None
        self.placeholder: Optional[Any] = None

    def gcs_client(
            self
    ) -> Client:
        return Client(
            project=self.attr.project_id,
            credentials=self.attr.cred_sa
        )

    def check_bucket(
            self
    ) -> bool:

        print(f"checking bucket {self.bucket_name}")

        try:
            self.client.get_bucket(bucket_or_name=self.bucket_name)
            print(f"bucket {self.bucket_name} found")
            return True

        except GoogleCloudNotFound:
            print(f"bucket {self.bucket_name} not found, creating new bucket")
            return False

    def get_bucket(
            self
    ) -> Bucket:

        if self.check_bucket():

            return Bucket(
                client=self.client,
                name=self.bucket_name
            )

        else:

            _bucket: Bucket = self.client.create_bucket(
                bucket_or_name=self.bucket_name,
                project=self.attr.project_id,
                location=self.attr.loc
            )

            print(f"Success! Bucket {self.bucket_name} succesfully created")

            return _bucket

    def create_folder(
            self,
            folder_name: str
    ) -> GoogleCloudStorage:

        if self.check_folder(folder_name=folder_name):

            print(f"{folder_name} already exists in bucket {self.bucket_name}")
            self.get_folder(folder_name=folder_name)

        else:

            _folder: Blob = self.bucket_ref.blob(f"{folder_name}/")
            _folder.upload_from_string("")

            self.placeholder = folder_name

            print(f"{_folder} created")

            self.folders.append(folder_name)

        return self

    def get_folder(
            self,
            folder_name: str
    ) -> GoogleCloudStorage:

        try:
            _folder_ref: Blob = self.bucket_ref.blob(f"{folder_name}/")

            self.placeholder = folder_name

        except GoogleCloudNotFound:
            print(f"{folder_name} is not found in bucket {self.bucket_name}")

        return self

    def list_all_folders(
            self
    ) -> List[str]:

        _blobs: Iterator = self.bucket_ref.list_blobs()

        _list_folders: List[str] = [
            each_blob
            for each_blob in _blobs
            if each_blob.name.endswith("/")
        ]

        return _list_folders

    def check_folder(
            self,
            folder_name: str
    ) -> bool:

        _list_folders: List[str] = self.list_all_folders()

        # Check if folder in the list
        _exists: bool = True if folder_name in _list_folders else False

        return _exists

    def upload_file(
            self,
            file_path: Union[os.PathLike, str] = "",
            folder_name: Optional[str] = None,
            blob_name: Optional[str] = None
    ) -> GoogleCloudStorage:

        # Check folder
        _folder_name: str = ""

        if folder_name is None:
            if self.placeholder is None:
                pass
            else:
                _folder_name = f"{self.placeholder}/"
        else:
            _folder_name = f"{folder_name}/"

        # Check blob_name
        _blob_name: str = ""

        if blob_name is None:
            _blob_name = os.path.basename(file_path)
        else:
            _blob_name = blob_name

        self.bucket_ref.blob(blob_name=f"{_folder_name}{_blob_name}").upload_from_file(file_obj=file_path)

        return self

    def count_blobs(
            self,
            folder_name: str
    ) -> int:

        _blobs: Iterator = self.bucket_ref.list_blobs(prefix=folder_name)

        _blobs: List[Blob] = list(_blobs)

        print(f"count blobs under {folder_name} folder is {len(_blobs)}")

        return len(_blobs)

    def download_blob(
            self,
            local_destination: Union[str, os.PathLike] = "",
            folder_name: Optional[str] = None,
            blob_name: Optional[str] = None
    ) -> GoogleCloudStorage:

        # Check folder
        _folder_name: str = ""

        if folder_name is None:
            if self.placeholder is None:
                pass
            else:
                _folder_name = f"{self.placeholder}/"
        else:
            _folder_name = f"{folder_name}/"

        # Check if local_destination exists
        def check_folder_exists(
                folder_path: Union[str, os.PathLike]
        ) -> bool:
            return os.path.isdir(folder_path)

        is_destination_exists: bool = check_folder_exists(folder_path=local_destination)

        if not is_destination_exists:
            os.makedirs(name=local_destination)

        with open(file=f"{local_destination}/{blob_name}", mode="wb") as file_obj:
            self.bucket_ref.blob(blob_name=f"{_folder_name}{blob_name}").download_to_file(file_obj=file_obj)

        return self
