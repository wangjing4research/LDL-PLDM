
import numpy as np
from LDLLDM import LDLLDM_Full
from LDM_SC import LDM_SC
from ldl_metrics import score
import multiprocessing 
from util import *    


def train(fold, train_x, train_y, test_x, test_y):
    
    #soft clustering process
    r = 10
    rho = 0.4
    l = 0.1
    
    
    print("soft clustering")
    ldm_sc = LDM_SC(train_y, r, rho, l)
    cluster, manifold = ldm_sc.solve()
    
    print("fitting models")
    l1 = 0.001
    l2 = 0 # without global label correlation
    l3 = 0.1
    ldlldm = LDLLDM_Full(train_x, train_y, l1, l2, l3, len(manifold), cluster, manifold)
    ldlldm.solve(200)
    
    val = score(test_y, ldlldm.predict(test_x))
    print(val)
    

def run_LDLPLDM(dataset):
    print(dataset)
    X, Y = np.load(dataset + "//feature.npy"), np.load(dataset + "//label.npy")
    train_inds = load_dict(dataset, "train_inds")
    test_inds = load_dict(dataset, "test_inds")
    
    for i in range(10):
        print('training ' + str(i) + ' fold')
        train_x, train_y = X[train_inds[i]], Y[train_inds[i]]
        test_x, test_y = X[test_inds[i]], Y[test_inds[i]]
        
        train(i, train_x, train_y, test_x, test_y)

        
if __name__ == "__main__":

    run_LDLPLDM("SJAFFE")
        

    
