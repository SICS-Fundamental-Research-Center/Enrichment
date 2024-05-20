#!/bin/bash

data=(
"person"
"imdb"
)

dataID=0
task="varying_deltaG"
outputFilePrefix="./results/varying_deltaG"

baseConfigFile='../dataset/'${data[${dataID}]}'/config.json'

mkdir ./configAll
mkdir ../results

outputConfigFile='./configAll/config_run.json'


for ratio in 0.05 0.1 0.15 0.2
do
# our method
# sampleRatio  IfIncEnrich  ifHEROPT  ifIncIE  sampleRatioTableBE  sampleRatioGraphBE  numOfWorks sampleRatioTableUpdateIE varying_m
python ../dataset/proc.py ${baseConfigFile} ${outputConfigFile} ${ratio} yes 1 1 1.0 1.0 20 0.1 5
    #hdfs dfs -rm '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #hdfs dfs -put ${outputConfigFile} '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #cd /root/project/enrich/discovery/
    cd ..
    ./run_enrich_unit.sh './shell/configAll/config_run.json' >> ${outputFilePrefix}'_'${data[${dataID}]}'_sampleRatio'${ratio}'_ours.txt'
    cd ./shell/
    #cd /root/project/enrich/discovery/enrich/shell
    # download results    
    #hdfs dfs -get ${outputFilePrefix}'_'${data[${dataID}]}'_sampleRatio'${ratio}'_ours.txt' ../results/
done
