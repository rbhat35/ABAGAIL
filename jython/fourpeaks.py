import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer

from array import array



"""
Commandline parameter(s):
   none
"""

N=200
T=N/5
fill = [2] * N
ranges = array('i', fill)

ef = FourPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

iterations = range(100, 50000, 100)
rhc_results = []
sa_results = []
ga_results = []

import time

rhc_startTime = time.time()
for iteration in iterations:
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, iteration)
    fit.train()
    rhc_optimalVal = ef.value(rhc.getOptimal())
    # print "RHC: " + str(rhc_optimalVal)
    rhc_results.append(rhc_optimalVal)
rhc_endTime = time.time()
print "RHC Completed in: " + str(rhc_endTime - rhc_startTime)

sa_startTime = time.time()
for iteration in iterations:
    sa = SimulatedAnnealing(1E11, .95, hcp)
    fit = FixedIterationTrainer(sa, iteration)
    fit.train()
    sa_optimalVal = ef.value(sa.getOptimal())
    # print "SA: " + str(sa_optimalVal)
    sa_results.append(sa_optimalVal)
sa_endTime = time.time()
print "SA Completed in: " + str(sa_endTime - sa_startTime)

ga_startTime = time.time()
for iteration in iterations:
    ga = StandardGeneticAlgorithm(200, 100, 10, gap)
    fit = FixedIterationTrainer(ga, iteration)
    fit.train()
    ga_optimalVal = ef.value(ga.getOptimal())
    # print "GA: " + str(ga_optimalVal)
    ga_results.append(ga_optimalVal)
ga_endTime = time.time()
print "GA Completed in: " + str(ga_endTime - ga_startTime)

import pickle
with open("fp_rhc.pickle", 'wb') as pfile:
    pickle.dump(rhc_results, pfile, pickle.HIGHEST_PROTOCOL)

with open("fp_sa.pickle", 'wb') as pfile:
    pickle.dump(sa_results, pfile, pickle.HIGHEST_PROTOCOL)

with open("fp_ga.pickle", 'wb') as pfile:
    pickle.dump(ga_results, pfile, pickle.HIGHEST_PROTOCOL)

with open("fp_iterations.pickle", 'wb') as pfile:
    pickle.dump(iterations, pfile, pickle.HIGHEST_PROTOCOL)



