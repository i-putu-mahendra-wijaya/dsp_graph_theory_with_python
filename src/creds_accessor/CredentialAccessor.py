from typing import Union
import os, yaml
from src.commons import EnvConstants as EnvConst
from src.creds_accessor.CredentialAccessorAttr import CredentialAccessorAttribute as CrAttr
from google.cloud import storage as st, bigquery as bq
from google.oauth2.service_account import Credentials as Cr


class CredentialAccessor():
    def __init__(
            self,
            env: str = EnvConst.DEV,
            on_server: bool = False,
            config_yaml_path: Union[str, os.PathLike] = "config.yaml",
            *args, **kwargs
    ) -> None:
        """ 
            Usage:
            To initialize all inheritable variables for Credential() class.
            
            Arguments:
            . env -> working environment options, "dev" (development) or "prod" (production)
            . on_server -> whether the access is in local or AWS server
            
            Info:
            . Last edited by: NICHOLAS DOMINIC <nicholas.dominic@agriaku.com>
            . Last edited at: January 13th, 2023
            . Copyright: DATA SCIENCE team, PT. Agriaku Digital Indonesia
        """

        super(CredentialAccessor, self).__init__()

        # get PROJECT_ROOT_DIR from config.yaml file
        with open(config_yaml_path, "r") as stream:
            try:
                loaded_stream: dict = yaml.safe_load(stream)
                self.ROOT_DIR = loaded_stream["PROJECT_ROOT_DIR"]
                self.job_location = loaded_stream["JOB_LOCATION"]
                self.project_id_dev = loaded_stream["PROJECT_ID_DEV"]
                self.project_id_prod = loaded_stream["PROJECT_ID_PROD"]
            except yaml.YAMLError as exc:
                print(exc)

        # working environment
        self.env = env
        self.on_server = on_server
        self.credentials_local_path = os.path.join(self.ROOT_DIR, "credentials")

        # location and project_id in BQ
        # self.job_location = "asia-southeast2" # Jakarta area
        # self.project_id_dev = "agriaku-dwh-dev-353603" # GCP Project ID - Development
        # self.project_id_prod = "disco-song-343611" # GCP Project ID - Production

        if self.on_server:
            # keys for server (both dev and prod)
            try:
                self.parent_dir = "/home/ubuntu/agriaku_dwh"  # from Variable.get("AGRIAKU_HOME")
                self.cred_path = self.parent_dir + "/keys/sa-gcp-agriaku.json"
            except Exception as e:
                print("[ERROR] {}".format(e))
        else:
            # keys for local
            self.cred_path = self.credentials_local_path + "/dev/" if self.env.upper() == "DEV" else self.credentials_local_path + "/prod/"
            self.cred_path += [c for c in os.listdir(self.cred_path) if c.endswith(".json")][0]

        # google oauth2 to read credential from JSON-formatted Service Account (sa) file
        self.google_auth_sa_credentials = Cr.from_service_account_file(filename=self.cred_path)

    def get_attr(
            self,
            *args, **kwargs
    ) -> CrAttr:
        """ 
            Usage:
            To get attributes from the class.
            
            Arguments:
            . None
        """

        return CrAttr(
            on_server=self.on_server,
            env=self.env,
            loc=self.job_location,
            project_id=self.project_id_dev if self.env.upper() == "DEV" else self.project_id_prod,
            cred_path=self.cred_path,
            cred_sa=self.google_auth_sa_credentials
        )

    def get_gbq_client(
            self,
            *args, **kwargs
    ) -> bq.client.Client:
        """ 
            Usage:
            To get a Google BigQuery client.
            
            Arguments:
            . None
        """

        attr = self.get_attr()
        return bq.Client(
            project=attr["project_id"],
            credentials=self.google_auth_sa_credentials
        )

    def get_gcs_client(
            self,
            *args, **kwargs
    ) -> st.client.Client:
        """ 
            Usage:
            To get a Google Cloud Storage client.
            
            Arguments:
            . None
        """

        attr = self.get_attr()
        return st.Client(
            project=attr["project_id"],
            credentials=self.google_auth_sa_credentials
        )
