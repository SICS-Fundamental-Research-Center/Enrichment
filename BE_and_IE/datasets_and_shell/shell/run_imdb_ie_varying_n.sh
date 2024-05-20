#!/bin/bash

data=(
"person"
"imdb"
)

dataID=1
task="varying_n"
outputFilePrefix="varying_n"

baseConfigFile='../dataset/'${data[${dataID}]}'/config.json'

mkdir ./configAll

outputConfigFile='./configAll/config_run.json'


for nw in 4 8 12 16 20 
do
    # our method
    # sampleRatio  IfIncEnrich  ifHEROPT  ifIncIE  sampleRatioTableBE  sampleRatioGraphBE  numOfWorks sampleRatioTableUpdateIE varying_m
python ../dataset/proc.py ${baseConfigFile} ${outputConfigFile} 0.1 yes 1 1 1.0 1.0 ${nw} 0.1 5
    hdfs dfs -rm '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    hdfs dfs -put ${outputConfigFile} '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    cd /root/project/enrich/discovery/
    ./run_enrich_unit.sh 'enrich_'${data[${dataID}]} 'enrich_'${data[${dataID}]} '/tmp/enrich/'${data[${dataID}]}'/config_run.json' ${outputFilePrefix}'_'${data[${dataID}]}'_nw'${nw}'_ours.txt'
    cd /root/project/enrich/discovery/enrich/shell
    # download results    
    hdfs dfs -get ${outputFilePrefix}'_'${data[${dataID}]}'_nw'${nw}'_ours.txt' ../results/

done


