import os
import yaml
from typing import Union, List, Any, Dict
from src.dao.GoogleBigquery import GoogleBigQuery
import pandas as pd


class ThriftyFetcher:

    def __init__(
            self,
            mygbq: GoogleBigQuery,
            sql_file_path: Union[str, os.PathLike],
            config_yaml_path: Union[str, os.PathLike] = "config.yaml",
            dataset_directory: Union[str, os.PathLike] = "./dataset/00_raw"
    ) -> None:
        """
        Check if query result already exists in folder datasets
        if already exists, then read corresponding CSV from folder datasets
        If not, then fetch from BigQuery

        Args:
            mygbq:
            sql_file_path:
        """
        with open(file=config_yaml_path, mode="r") as cfg_yaml:
            try:
                cfg: Dict[str, str] = yaml.safe_load(
                    stream=cfg_yaml
                )
            except yaml.YAMLError as exc:
                print(exc)

        self.project_root_dir: str = cfg["PROJECT_ROOT_DIR"]
        self.mygbq: GoogleBigQuery = mygbq
        self.sql_file_path: Union[str, os.PathLike] = sql_file_path
        self.datasets_dir: Union[str, os.PathLike] = dataset_directory

        __file_path: Any = os.path.normpath(self.sql_file_path)
        __path_list: List[str] = __file_path.split(os.sep)
        __query_file_name: str = __path_list[-1]
        __query_file_name, _ = os.path.splitext(__query_file_name)

        self.query_file_name: str = __query_file_name
        self.fetched_csv_filename: str = f"fetched_{self.query_file_name}.csv"
        self.fetched_csv_file_path: Union[str, os.PathLike] = os.path.join(
            self.project_root_dir,
            self.datasets_dir,
            self.fetched_csv_filename
        )

        with open(file=self.sql_file_path, mode="r") as file_ref:
            query_string: str = file_ref.read()

        self.query_string: str = query_string

    def get_data(
            self
            , check_local: bool = True
    ) -> pd.DataFrame:
        """
        :param check_local: A boolean flag indicating whether to check for local data or not. Defaults to True.
        :return: Returns a pandas DataFrame containing the query result.
        """

        query_result: pd.DataFrame

        if not check_local:
            if not os.path.exists(self.fetched_csv_file_path):
                print("fetch from GBQ")
                query_result = self.mygbq.gbq_read(
                    query=self.query_string
                )
                query_result.to_csv(
                    path_or_buf=self.fetched_csv_file_path,
                    sep=","
                )
        else:
            if not os.path.exists(self.fetched_csv_file_path):
                print("fetch from GBQ")
                query_result = self.mygbq.gbq_read(
                    query=self.query_string
                )
                query_result.to_csv(
                    path_or_buf=self.fetched_csv_file_path,
                    sep=","
                )
            else:
                print(f"fetch from {self.fetched_csv_file_path}")
                query_result = pd.read_csv(
                    filepath_or_buffer=self.fetched_csv_file_path
                )

        return query_result
