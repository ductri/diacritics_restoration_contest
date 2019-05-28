#!/bin/bash

docker stop trind_dia
sleep 5

./docker/jupyter.sh
