#!/bin/bash

data=(
"person"
"imdb"
)

dataID=0
task="varying_m"
outputFilePrefix="./results/varying_m"

baseConfigFile='../dataset/'${data[${dataID}]}'/config.json'

mkdir ./configAll
mkdir ../results

outputConfigFile='./configAll/config_run.json'


for ms in 1 2 3 4 5
do
    # our method
    # sampleRatio  IfIncEnrich  ifHEROPT  ifIncIE  sampleRatioTableBE  sampleRatioGraphBE  numOfWorks sampleRatioTableUpdateIE varying_m
python ../dataset/proc.py ${baseConfigFile} ${outputConfigFile} 1.0 no 1 1 1.0 1.0 20 0.1 ${ms}
    #hdfs dfs -rm '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #hdfs dfs -put ${outputConfigFile} '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #cd /root/project/enrich/discovery/
    cd ..
    ./run_enrich_unit.sh './shell/configAll/config_run.json' >> ${outputFilePrefix}'_'${data[${dataID}]}'_m'${ms}'_ours.txt'
    cd ./shell/
    #cd /root/project/enrich/discovery/enrich/shell
    # download results    
    #hdfs dfs -get ${outputFilePrefix}'_'${data[${dataID}]}'_m'${ms}'_ours.txt' ../results/

    # baseline
    # sampleRatio  IfIncEnrich  ifHEROPT  ifIncIE  sampleRatioTableBE  sampleRatioGraphBE  numOfWorks
    #python ../dataset/proc.py ${baseConfigFile} ${outputConfigFile} 1.0 no 0 1 1.0 1.0 ${nw}
    #hdfs dfs -rm '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #hdfs dfs -put ${outputConfigFile} '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #cd /root/project/enrich/discovery/
    #./run_enrich_unit.sh 'enrich_'${data[${dataID}]} 'enrich_'${data[${dataID}]} '/tmp/enrich/'${data[${dataID}]}'/config_run.json' ${outputFilePrefix}'_'${data[${dataID}]}'_nw'${nw}'_others.txt'
    #cd /root/project/enrich/discovery/enrich/shell
    # download results    
    #hdfs dfs -get ${outputFilePrefix}'_'${data[${dataID}]}'_nw'${nw}'_others.txt' ../results/

done


