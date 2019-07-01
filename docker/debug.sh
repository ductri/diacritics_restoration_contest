#!/bin/bash

docker run -ti --runtime=nvidia --rm -e PYTHONIOENCODING=utf-8 \
-v `pwd`/source:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/train.sh"
