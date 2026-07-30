#!/bin/sh
set -eu
export PYTHONPATH="/app/src"
# The add-on's CPU-heavy work is deliberately lower priority than Home
# Assistant core and interactive services on the same host.
exec nice -n 10 python -m farmbot_vision
