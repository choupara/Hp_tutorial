#!/bin/bash -l
# execution command is "bash run.sh"
tstamp=`date '+%Y_%m_%d_%H_%M_%S'`
log_file="log_${tstamp}.txt"

#!/bin/sh

i=1
while [ "$i" -le 5 ]; do
    echo "Run Number $i"
    #out-layer1-to-2
    max=600
    min=400
    x_0=$(($RANDOM%($max-$min+1)+$min))
    x_1=$(($RANDOM%($max-$min+1)+$min))

    #batchsize
    max=60
    min=40
    x_2=$(($RANDOM%($max-$min+1)+$min))

    #epochs
    max=10
    min=5
    x_3=$(($RANDOM%($max-$min+1)+$min))

    echo "Parameter --out-layer1=($x_0) --out-layer2=($x_1) --batchsize=($x_2) --epochs=($x_3)" >> $log_file 2>&1
    sh wrapping_script.sh --out-layer1=$x_0 --out-layer2=$x_1 --batchsize=$x_2 --epochs=$x_3 > output.txt  2>&1
    grep "RESULT:" output.txt | tail -1  >>  $log_file 2>&1 
    i=$(( $i + 1 ))
done
