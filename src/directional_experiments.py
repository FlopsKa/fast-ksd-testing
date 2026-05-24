import numpy as np
import pandas as pd
from tqdm import tqdm

from nystroem_ksd.ksd import DirectionalKSD, TestBenchmark, NystroemDirectionalKSD, Rayleigh, ContextTimer, UniformCircle, VonMises, VonMisesFisher3d,  UniformSphere3d

def experiments_d2(level_alpha,repetitions,ns=[30,50,100,200]):
    ## d=2 
    optim_kappa_param = 0.12 # obtained by trying different kappas and using the one maximizing power
    data = {
        "Uniform" : UniformCircle(),
        "VonMises" : VonMises()
        }

    tests = {
        "Rayleigh" : Rayleigh(level_alpha=level_alpha),
        "N-KSD" : NystroemDirectionalKSD(kappa=optim_kappa_param, d=2, level_alpha=level_alpha),
        "KSD" : DirectionalKSD(kappa=optim_kappa_param, d=2, level_alpha=level_alpha)
        }

    res = pd.DataFrame()
    for n in tqdm(ns):
        for data_key, data_value in data.items():
            for test_key, test_value in tests.items():
                with ContextTimer() as t:
                    rejects = TestBenchmark(stat_test=test_value).error(n=n,data_gen=data_value.gen, repetitions=repetitions)
                res = pd.concat((res,pd.DataFrame({
                    "n" : [n],
                    "data" : [data_key],
                    "test" : [test_key],
                    "rejects" : [rejects],
                    "time" : [t.secs]
                })))
    print("Results for d=2:")
    print(res)
    res.to_csv("./results/directional-d2.csv")

def experiments_d3_uniform(level_alpha,repetitions,optim_kappa_param=0.28,ns=[30,50,100,200]):
    res = pd.DataFrame()
    data = { "Uniform" : UniformSphere3d() }
    tests = {
            "N-KSD" : NystroemDirectionalKSD(kappa=optim_kappa_param, d=3, level_alpha=level_alpha),
            "KSD" : DirectionalKSD(kappa=optim_kappa_param, d=3, level_alpha=level_alpha)
            }

    for n in tqdm(ns):
        for data_key, data_value in data.items():
            for test_key, test_value in tests.items():
                with ContextTimer() as t:
                    rejects = TestBenchmark(stat_test=test_value).error(n=n,data_gen=data_value.gen, repetitions=repetitions)
                res = pd.concat((res,pd.DataFrame({
                    "n" : [n],
                    "kappa" : [0],
                    "data" : [data_key],
                    "test" : [test_key],
                    "rejects" : [rejects],
                    "time" : [t.secs]
                })))
    print("Results for d=3 (Uniform):")
    print(res)
    res.to_csv("./results/directional-d3-uniform.csv")

def experiments_d3_vMF(level_alpha,repetitions,optim_kappa_param=0.28,n=300,kappas=np.linspace(0.01,6,10)):
    res = pd.DataFrame()
    data = { "VonMises" : VonMisesFisher3d }
    tests = {
            "N-KSD" : NystroemDirectionalKSD(kappa=optim_kappa_param, d=3, level_alpha=level_alpha),
            "KSD" : DirectionalKSD(kappa=optim_kappa_param, d=3, level_alpha=level_alpha)
            }

    for kappa in tqdm(kappas):
        for data_key, data_value in data.items():
            for test_key, test_value in tests.items():
                with ContextTimer() as t:
                    rejects = TestBenchmark(stat_test=test_value).error(n=n,data_gen=data_value(kappa=kappa).gen, repetitions=repetitions)
                res = pd.concat((res,pd.DataFrame({
                    "n" : [n],
                    "kappa" : [kappa],
                    "data" : [data_key],
                    "test" : [test_key],
                    "rejects" : [rejects],
                    "time" : [t.secs]
                })))

    print("Results for d=3 (vMF):")
    print(res)
    res.to_csv("./results/directional-d3-vMF.csv")

def main():    
    level_alpha = 0.01
    ns = [30,50,100,200]
    kappas=np.linspace(0.01,6,10)
    repetitions = 600

    experiments_d2(level_alpha=level_alpha,repetitions=repetitions,ns=ns)
    experiments_d3_uniform(level_alpha=level_alpha,repetitions=repetitions,ns=ns)
    experiments_d3_vMF(level_alpha=level_alpha,repetitions=repetitions,n=200,kappas=kappas)

        
if __name__=="__main__":
    main()
