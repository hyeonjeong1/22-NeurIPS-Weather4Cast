#!/bin/bash
for W in 1 0.8 0.6 0.4 0.2;
do
  export WEIGHT=$W
  for G in 1 2 3 4 5
  do
    export GAMMA=$G
    envsubst < models/configurations/config_15_.yaml
    envsubst < models/configurations/config_34_.yaml
    envsubst < models/configurations/config_76_.yaml
    bash script.sh
  done
done