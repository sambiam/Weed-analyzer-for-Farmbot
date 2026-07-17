#!/bin/sh
set -eu
export PYTHONPATH="/app/src"
exec python -m farmbot_vision
