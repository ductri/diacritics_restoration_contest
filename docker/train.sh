#!/bin/bash

docker logs -f --timestamps $(docker run --runtime=nvidia -d -e PYTHONIOENCODING=utf-8 --name="diacritics_$(date +"%y-%m-%d_%H_%M_%S")" \
-v `pwd`/source:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/train.sh")
