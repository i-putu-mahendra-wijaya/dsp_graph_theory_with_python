import traceback

import pandas as pd

from src.commons import EnvConstants as EnvConst
from src.commons.Logger import Logger
from src.commons.ThriftyFetcher import ThriftyFetcher
from src.creds_accessor.CredentialAccessor import CredentialAccessor as CrAcc
from src.dao.GoogleBigquery import GoogleBigQuery


if __name__ == "__main__":

    cred_accessor: CrAcc = CrAcc(
        env=EnvConst.PROD,
        on_server=False
    )

    mygbq: GoogleBigQuery = GoogleBigQuery(
        attr=cred_accessor.get_attr()
    )

    logger: Logger = Logger(
        logger_name="default_logger_name",
        caller_name=__name__,
        attr=cred_accessor.get_attr(),
        level=Logger.INFO
    )

    try:
        fetcher: ThriftyFetcher = ThriftyFetcher(
            mygbq=mygbq,
            sql_file_path="queries/your_query_file.sql",
            dataset_directory="./datasets"
        )

        data_df: pd.DataFrame = fetcher.get_data()

    except Exception as e:

        logger.error(
            message= \
            f"""
            Exception occurred: {traceback.format_exc()}
            \n\n
            """
        )