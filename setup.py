import os
import shutil

import platformdirs
from setuptools import setup


# HACK: This is a hack to create config files in the user's config directory
def _post_install() -> None:
    conf_dir = platformdirs.user_config_dir('pmtrendviz')
    os.makedirs(conf_dir, exist_ok=True)
    shutil.copy('elasticsearch/es.env', os.path.join(conf_dir, 'es.env'))


setup()
_post_install()
