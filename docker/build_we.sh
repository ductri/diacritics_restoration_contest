#!/bin/bash

docker run --rm --runtime=nvidia -ti -e PYTHONIOENCODING=utf-8 --name="diacritics_we" \
-v `pwd`/source:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/build_we.sh"
