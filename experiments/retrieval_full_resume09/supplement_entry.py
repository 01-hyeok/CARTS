import sys
import gate
sys.path.insert(0,str(gate.VALIDATED))
import runtime_all_channels
sys.path.insert(0,str(gate.ROOT/'experiments/retrieval_diagnostic_supplement01'))
import supplement_inference
if __name__=='__main__':
    gate.verify();assert sys.argv[1:]==['FULL','Weather_720']
    supplement_inference.run('FULL','Weather_720')
