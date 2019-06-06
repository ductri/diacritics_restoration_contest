#!/bin/bash

docker logs -f --timestamps $(docker run --runtime=nvidia -d -e PYTHONIOENCODING=utf-8 --name="diacritics_tensorboard" \
-v `pwd`/source:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/tensorboard.sh")
