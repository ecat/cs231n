#/usr/bin/bash

for ii in {1..2}
do
    mkdir -p "data/scan$ii"
    unzip "P$ii.zip" -d "data/scan$ii"
done