"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying winequality-red.

Based on AbaloneTest.java by Hannah Lau
"""

from __future__ import with_statement

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

INPUT_FILE = os.path.join("diabetes.csv")

INPUT_LAYER = 11
HIDDEN_LAYER = 5
OUTPUT_LAYER = 1


def initialize_instances():
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []

    # df = pd.read_csv('winequality-red.csv')
    # labels = df.pop('quality').values

    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(df)
    # df = pd.DataFrame(x_scaled, columns=list(df))
    # df = pd.concat([df, labels], axis = 1)

    # Read in the abalone.txt CSV file
    with open(INPUT_FILE, "r") as abalone:
        reader = list(csv.reader(abalone))

    import random
    import math
    random.shuffle(reader)
    for ind in xrange(len(reader[0]) - 1):
        vals = [float(row[ind]) for row in reader]

        mean = sum(vals)/len(vals)
        variance = sum([val ** 2 for val in vals])/float(len(vals)) - mean ** 2

        for row in xrange(len(reader)):
            reader[row][ind] = math.ceil((vals[row] - mean)/(variance ** 0.5) *1000)/1000

    for row in reader:
        instance = Instance([float(value) for value in row[:-1]])
        instance.setLabel(Instance(float(row[-1])))
        instance.setLabel(Instance(0 if float(row[-1]) == 0.0 else 1))
        instances.append(instance)

    trainingInstances = instances[:int(len(instances)*0.7)]
    testingInstances = instances[int(len(instances)*0.7):]
    print "Number of training instances: " + str(len(trainingInstances))
    print "Number of testing instances: " + str(len(testingInstances))
    return trainingInstances, testingInstances


def train(oa, network, oaName, instances, measure, TRAINING_ITERATIONS=2000):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    # print "\nError results for %s\n---------------------------" % (oaName,)

    for iteration in xrange(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        # print "%0.03f" % error


def main():
    """Run algorithms on the winequality-red dataset."""

    learningCurve_data = {}
    learning_curve_file = "NN_learning.pickle"

    numIterations_data = {}
    num_iterations_file = "NN_iterations.pickle"

    trainingInstances, testingInstances = initialize_instances()
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(trainingInstances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA"]

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
    oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    learningCurve_size = xrange(20, len(trainingInstances), 80)
    numIterations_iters = xrange(100, 5000, 150)

    for i, name in enumerate(oa_names):
        start = time.time()
        correct = 0
        incorrect = 0

        train(oa[i], networks[i], oa_names[i], trainingInstances, measure)
        end = time.time()
        training_time = end - start
        print "\nTraining time: %0.03f seconds" % (training_time,)

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        start = time.time()
        for instance in trainingInstances:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            predicted = instance.getLabel().getContinuous()
            # print networks[i].getOutputValues()
            actual = networks[i].getOutputValues().get(0)

            # print predicted
            # print actual

            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1

        end = time.time()
        testing_time = end - start

        print "\nTRAINING: Results for %s: \nCorrectly classified %d instances." % (name, correct)
        print "\nTRAINING: Incorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        print "\nTRAINING: Testing time: %0.03f seconds\n" % (testing_time,)

        correct = 0
        incorrect = 0
        start = time.time()
        for instance in testingInstances:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            predicted = instance.getLabel().getContinuous()
            actual = networks[i].getOutputValues().get(0)

            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1

        end = time.time()
        testing_time = end - start

        print "\nTESTING: Results for %s: \nCorrectly classified %d instances." % (
            name, correct)
        print "\nTESTING: Incorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (
            incorrect, float(correct)/(correct+incorrect)*100.0)
        print "\nTESTING: Testing time: %0.03f seconds\n" % (testing_time,)



        trainAccuracy = []
        testAccuracy = []

        for num in learningCurve_size:
            data_set = DataSet(trainingInstances[:num])
            networks = []  # BackPropagationNetwork
            nnop = []  # NeuralNetworkOptimizationProblem
            oa = []  # OptimizationAlgorithm

            for name in oa_names:
                classification_network = factory.createClassificationNetwork(
                    [INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
                networks.append(classification_network)
                nnop.append(NeuralNetworkOptimizationProblem(
                    data_set, classification_network, measure))

            oa.append(RandomizedHillClimbing(nnop[0]))
            oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
            oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

            train(oa[i], networks[i], oa_names[i], trainingInstances, measure)
            optimal_instance = oa[i].getOptimal()
            networks[i].setWeights(optimal_instance.getData())

            correct = 0
            incorrect = 0
            for instance in trainingInstances[:num]:
                networks[i].setInputValues(instance.getData())
                networks[i].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[i].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1


            trainAccuracy.append(float(correct)/(correct+incorrect)*100.0)

            correct = 0
            incorrect = 0
            for instance in testingInstances:
                networks[i].setInputValues(instance.getData())
                networks[i].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[i].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1
            testAccuracy.append(float(correct)/(correct+incorrect)*100.0)

        learningCurve_data[oa_names[i]] = [trainAccuracy, testAccuracy]


        trainAccuracy = []
        testAccuracy = []

        for num in numIterations_iters:
            data_set = DataSet(trainingInstances)
            networks = []  # BackPropagationNetwork
            nnop = []  # NeuralNetworkOptimizationProblem
            oa = []  # OptimizationAlgorithm

            for name in oa_names:
                classification_network = factory.createClassificationNetwork(
                    [INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
                networks.append(classification_network)
                nnop.append(NeuralNetworkOptimizationProblem(
                    data_set, classification_network, measure))

            oa.append(RandomizedHillClimbing(nnop[0]))
            oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
            oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

            train(oa[i], networks[i], oa_names[i], trainingInstances, measure, TRAINING_ITERATIONS = num)
            optimal_instance = oa[i].getOptimal()
            networks[i].setWeights(optimal_instance.getData())

            correct = 0
            incorrect = 0
            for instance in trainingInstances:
                networks[i].setInputValues(instance.getData())
                networks[i].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[i].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1
            trainAccuracy.append(float(correct)/(correct+incorrect)*100.0)

            correct = 0
            incorrect = 0
            for instance in testingInstances:
                networks[i].setInputValues(instance.getData())
                networks[i].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[i].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1
            testAccuracy.append(float(correct)/(correct+incorrect)*100.0)

        numIterations_data[oa_names[i]] = [trainAccuracy, testAccuracy]

        print "------------------------------------------------------------"

    with open("NN_learningCurveAccuracy.pickle", 'wb') as file:
        pickle.dump(learningCurve_data, file, pickle.HIGHEST_PROTOCOL)

    with open("NN_numIterationsAccuracy.pickle", 'wb') as file:
        pickle.dump(numIterations_data, file, pickle.HIGHEST_PROTOCOL)

    with open("NN_learningCurveSize.pickle", 'wb') as file:
        pickle.dump(learningCurve_size, file, pickle.HIGHEST_PROTOCOL)

    with open("NN_numIters.pickle", 'wb') as file:
        pickle.dump(numIterations_iters, file, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

