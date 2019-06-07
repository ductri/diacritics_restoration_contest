#!/bin/bash

docker run -ti --rm --runtime=nvidia -e PYTHONIOENCODING=utf-8 --name="diacritics_terminal" \
-v `pwd`/source:/source \
ductricse/pytorch /bin/bash
