# Central runtime configuration.
# All sensitive values are loaded from environment variables so the project can
# be published safely without hardcoding secrets in source control.
#
# This file is part of KhunDech.
# KhunDech is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os


DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
ALLOWED_USER_ID = int(os.environ.get("ALLOWED_USER_ID", "540111138323693568"))
MAIN_CHANNEL_ID = int(os.environ.get("MAIN_CHANNEL_ID", "1482618900347883590"))
CRON_CHANNEL_ID = int(os.environ.get("CRON_CHANNEL_ID", "1482618900347883590"))
LEARNING_CHANNEL_ID = int(os.environ.get("LEARNING_CHANNEL_ID", os.environ.get("CRON_CHANNEL_ID", "1482618900347883590")))

_raw_model = os.environ.get("AI_MODEL", "gemini-3.1-flash-lite-preview")
AI_MODEL = _raw_model.removeprefix("google/")

DATA_DIR = "/app/data"
SOURCE_DIR = "/app"