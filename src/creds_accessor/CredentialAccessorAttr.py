import os
from typing import Union
from src.commons import EnvConstants as EnvConst


class CredentialAccessorAttribute:

    def __init__(
            self,
            on_server: bool,
            env: EnvConst,
            loc: str,
            project_id: str,
            cred_path: Union[str, os.PathLike],
            cred_sa: str
    ) -> None:
        """
        Usage:
            . Template to wrap return value from credential_accessory.attribute

        Arguments:
             . None

        Info:
        . Last edited by: I PUTU MAHENDRA WIJAYA <i.wijaya@agriaku.com>
        . Last edited at: March 04, 2023
        . Copyright: DATA SCIENCE team, PT. Agriaku Digital Indonesia
        """

        super(CredentialAccessorAttribute, self).__init__()

        self.on_server: bool = on_server
        self.env: EnvConst = env
        self.loc: str = loc
        self.project_id: str = project_id
        self.cred_path: Union[str, os.PathLike] = cred_path
        self.cred_sa: str = cred_sa
