#!/bin/bash

data=(
"person"
"imdb"
)

dataID=1
task="varying_deltaD"
outputFilePrefix="./results/varying_deltaD"

baseConfigFile='../dataset/'${data[${dataID}]}'/config.json'

mkdir ./configAll
mkdir ../results

outputConfigFile='./configAll/config_run.json'


for ratio in 0.05 0.1 0.15 0.2 
do
    # our method
    # sampleRatio  IfIncEnrich  ifHEROPT  ifIncIE  sampleRatioTableBE  sampleRatioGraphBE  numOfWorks sampleRatioTableUpdateIE varying_m
python ../dataset/proc.py ${baseConfigFile} ${outputConfigFile} 0.1 yes 1 1 1.0 1.0 20 ${ratio} 5
    #hdfs dfs -rm '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #hdfs dfs -put ${outputConfigFile} '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #cd /root/project/enrich/discovery/
    cd ..
    ./run_enrich_unit.sh './shell/configAll/config_run.json' >> ${outputFilePrefix}'_'${data[${dataID}]}'_ratio'${ratio}'_ours.txt'
    cd ./shell/
    #cd /root/project/enrich/discovery/enrich/shell
    # download results    
    #hdfs dfs -get ${outputFilePrefix}'_'${data[${dataID}]}'_ratio'${ratio}'_ours.txt' ../results/

    # baseline
    # sampleRatio  IfIncEnrich  ifHEROPT  ifIncIE  sampleRatioTableBE  sampleRatioGraphBE  numOfWorks
    #python ../dataset/proc.py ${baseConfigFile} ${outputConfigFile} 1.0 no 0 1 ${ratio} 1.0 20
    #hdfs dfs -rm '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #hdfs dfs -put ${outputConfigFile} '/tmp/enrich/'${data[${dataID}]}'/config_run.json'
    #cd /root/project/enrich/discovery/
    #./run_enrich_unit.sh 'enrich_'${data[${dataID}]} 'enrich_'${data[${dataID}]} '/tmp/enrich/'${data[${dataID}]}'/config_run.json' ${outputFilePrefix}'_'${data[${dataID}]}'_ratio'${ratio}'_others.txt'
    #cd /root/project/enrich/discovery/enrich/shell
    # download results    
    #hdfs dfs -get ${outputFilePrefix}'_'${data[${dataID}]}'_ratio'${ratio}'_others.txt' ../results/

done


