#/usr/bin/bash

for ii in {8..10}
do
    mkdir -p "data/scan$ii"
    unzip "P$ii.zip" -d "data/scan$ii"
done
