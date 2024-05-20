import json
import sys

config_file = sys.argv[1]
output_file = sys.argv[2]

# all changed parameters
# 1. For IE: sample ratio of graph updates
sampleRatio = sys.argv[3]
# 2. For IE: whether we need to do incremental enrichment
ifIncEnrich = sys.argv[4]
# 3. For BE: whether we use optimized HER
ifHEROPT = int(sys.argv[5])
# 4. For IE: whether to use incremental enrichment or full run (batch) for IE
ifIncIE = int(sys.argv[6])
# 5. For BE: precentage of table
sampleRatioTableBE = sys.argv[7]
# 6. For BE: precentage of graph
sampleRatioGraphBE = sys.argv[8]
# 7. For BE and IE: the number of workers
numOfWorks = int(sys.argv[9])
# 8. For IE: sample ratio of table updates
sampleRatioTableUpdateIE = sys.argv[10]
# 9. For varying m: the number of attributes to be populated
m = int(sys.argv[11])

config = None
with open(config_file, 'r') as f:
    config = json.load(f)


config['sampleRatio'] = sampleRatio
config['ifIncEnrich'] = ifIncEnrich
config['ifHEROPT'] = ifHEROPT
config['ifIncIE'] = ifIncIE
config['sampleRatioTableBE'] = sampleRatioTableBE
config['sampleRatioGraphBE'] = sampleRatioGraphBE
config['numOfWorkers'] = numOfWorks
config['sampleRatioTableUpdateIE'] = sampleRatioTableUpdateIE

ES = config['enrichedSchemas']
config['enrichSchemas'] = ES[:m]


with open(output_file, 'w') as b:
    json.dump(config, b)


