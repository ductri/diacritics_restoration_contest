#!/bin/bash

echo "Start reverse  ssh ..."
service ssh start

ssh -R 9998:localhost:22 root@213.246.38.101

#ssh -NR 5057:localhost:6006 root@213.246.38.101 &
