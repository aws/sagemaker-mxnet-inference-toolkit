#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    # shift 1
    # mxnet-model-server --start --mms-config /home/model-server/config.properties
    python -c "from sagemaker_mxnet_serving_container import serving; serving.main()"
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
