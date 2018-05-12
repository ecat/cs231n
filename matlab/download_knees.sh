#/usr/bin/bash

for ii in {1..10}
do
    DOWNLOAD_URL="http://mridata.org/knees/fully_sampled/p$ii/e1/s1/P$ii.zip"
    echo $DOWNLOAD_URL
    wget $DOWNLOAD_URL
done