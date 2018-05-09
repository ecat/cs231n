#/usr/bin/bash

for ii in {1..2}
do
    DOWNLOAD_URL="http://mridata.org/knees/fully_sampled/p$ii/e1/s1/P$ii.zip"
    echo $DOWNLOAD_URL
    wget $DOWNLOAD_URL -q # -q turns off output
done