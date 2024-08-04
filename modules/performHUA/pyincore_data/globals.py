# Copyright (c) 2021 University of Illinois and others. All rights reserved.  # noqa: D100, INP001
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import logging
import os
from logging import config as logging_config

PACKAGE_VERSION = '0.3.0'

PYINCORE_DATA_ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))  # noqa: PTH120

LOGGING_CONFIG = os.path.abspath(  # noqa: PTH100
    os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logging.ini')  # noqa: PTH100, PTH118, PTH120
)
logging_config.fileConfig(LOGGING_CONFIG)
LOGGER = logging.getLogger('pyincore-data')
