#!/bin/bash

docker run -ti --runtime=nvidia -e PYTHONIOENCODING=utf-8 --rm \
-v `pwd`/source:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/run_unittests.sh"
