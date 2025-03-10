import logging
import os
import yaml
from typing import Union

from google.cloud import storage as st

import src.commons.EnvConstants as EnvConst
from src.creds_accessor.CredentialAccessorAttr import CredentialAccessorAttribute as CrAttr
from src.dao.GoogleCloudStorage import GoogleCloudStorage

# Define own type
LoggingLevel = int


class Logger:
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG

    def __init__(
            self,
            logger_name: str,
            caller_name: str,
            attr: CrAttr,
            gcs_client: GoogleCloudStorage = None,
            is_save_to_gcs: bool = False,
            level: LoggingLevel = logging.INFO
    ) -> None:
        """
            Usage:
            To  setup and write log

            Arguments:
                . attr -> credential accessor attribute object
                . gcs_client -> google cloud storage client object
                . level -> set the lowest logging level that will be recorded. Allowed options:
                                Logger.CRITICAL,
                                Logger.ERROR,
                                Logger.WARNING,
                                Logger.INFO,
                                Logger.DEBUG

            Info:
            . Last edited by: I PUTU MAHENDRA WIJAYA <i.wijaya@agriaku.com>
            . Last edited at: Mar 04, 2023
            . Copyright: DATA SCIENCE team, PT. Agriaku Digital Indonesia

        """
        super(Logger, self).__init__()

        # Check if is_save_to_gcs is True, then gcs client must be supplied
        try:
            if is_save_to_gcs and gcs_client is None:
                raise ValueError("is_save_to_gcs is set to True, but gcs_client is missing")

        except ValueError as e:
            print(e)

        # Check if env is PROD then gcs_client must be supplied
        try:
            if attr.env == EnvConst.PROD and gcs_client is None:
                raise ValueError("PROD env is detected but gcs_client is still 'None'")
        except ValueError as e:
            print(e)

        # Get project ROOT_DIR from config.yaml.jinja
        with open("config.yaml", "r") as stream:
            try:
                self.ROOT_DIR = yaml.safe_load(stream)["PROJECT_ROOT_DIR"]
            except yaml.YAMLError as exc:
                print(exc)

        self.logger_name: str = logger_name
        self.caller_name: str = caller_name

        self.on_server: bool = attr.on_server
        self.env: EnvConst = attr.env
        self.loc: str = attr.loc
        self.project_id: str = attr.project_id
        self.cred_path: Union[str, os.PathLike] = attr.cred_path
        self.cred_sa: str = attr.cred_sa

        self.gcs_client: GoogleCloudStorage = gcs_client
        self.is_save_to_gcs: bool = is_save_to_gcs

        normalized_path_: Union[str, os.PathLike] = os.path.normpath(self.ROOT_DIR)  # ensure path complies with the OS
        normalized_path_comp_: list = normalized_path_.split(os.sep)

        self.project_name: str = normalized_path_comp_[len(normalized_path_comp_) - 1]

        self.logging_level: LoggingLevel = level

        self.LOG_DIR: Union[str, os.PathLike] = os.path.join(self.ROOT_DIR, "logging")

        self.log_filename: Union[str, os.PathLike] = os.path.join(
            self.LOG_DIR, f"{self.project_name}_{self.logger_name}.log"
        )

        formatter: logging.Formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

        self.logger = logging.getLogger(f"{self.project_name}_{self.logger_name}_{self.caller_name}")

        self.logger.setLevel(self.logging_level)

        logger_file_handler_ = logging.FileHandler(
            filename=self.log_filename,
            mode="a"  # writing mode
        )
        logger_file_handler_.setFormatter(formatter)
        logger_file_handler_.setLevel(self.logging_level)

        logger_stream_handler_ = logging.StreamHandler()
        logger_stream_handler_.setFormatter(formatter)
        logger_stream_handler_.setLevel(self.logging_level)

        self.logger.addHandler(logger_file_handler_)
        self.logger.addHandler(logger_stream_handler_)

    def upload_log_to_gcs(
            self
    ) -> None:
        """
        Usage:
        . Upload log from LOG_DIR to GCS

        Argument:
        . None

        Return:
        . None
        """

        self.gcs_client.upload_blob_from_file(
            source_file_path=self.log_filename
        )

    def critical(self, message: str) -> None:
        self.logger.critical(message)

        if self.is_save_to_gcs:
            self.upload_log_to_gcs()

    def error(self, message: str) -> None:
        self.logger.error(message)

        if self.is_save_to_gcs:
            self.upload_log_to_gcs()

    def warning(self, message: str) -> None:
        self.logger.warning(message)

        if self.is_save_to_gcs:
            self.upload_log_to_gcs()

    def info(self, message: str) -> None:
        self.logger.info(message)

        if self.is_save_to_gcs:
            self.upload_log_to_gcs()

    def debug(self, message: str) -> None:
        self.logger.debug(message)

        if self.is_save_to_gcs:
            self.upload_log_to_gcs()
