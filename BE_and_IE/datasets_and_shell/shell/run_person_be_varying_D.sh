#!/bin/bash

data=(
"person"
"imdb"
)

dataID=0
task="varying_D"
outputFilePrefix="./results/varying_D"

baseConfigFile='../dataset/'${data[${dataID}]}'/config.json'

mkdir ./configAll
mkdir ../results

outputConfigFile='./configAll/config_run.json'


for ratio in 0.2 0.4 0.6 0.8 1.0 
do
    # our method
    # sampleRatio  IfIncEnrich  ifHEROPT  ifIncIE  sampleRatioTableBE  sampleRatioGraphBE  numOfWorks sampleRatioTableUpdateIE varying_m
python ../dataset/proc.py ${baseConfigFile} ${outputConfigFile} 1.0 no 1 1 ${ratio} 1.0 20 0.1 5
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


