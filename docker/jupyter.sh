#!/bin/bash

docker logs -f --timestamps $(docker run --runtime=nvidia -d -e PYTHONIOENCODING=utf-8 --name=trind_dia --rm \
-v `pwd`/source/:/source \
-v `pwd`/../dataset/vietnamese_tone_prediction:/dataset/vietnamese_tone_prediction \
-v `pwd`/../dataset/wiki/:/dataset/wiki:ro \
-p 5050:5050 \
ductricse/pytorch /bin/bash -c "jupyter notebook --port=5050 --allow-root --ip=0.0.0.0")
