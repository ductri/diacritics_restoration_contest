#!/bin/bash

docker run -ti --rm -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/web \
-p 2609:8080 \
halverneus/static-file-server:latest
