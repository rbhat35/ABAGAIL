import matplotlib.pyplot as plt
import pickle

def tsp():
    for pre in ["", "_1000", "_5000"]:
        with open('tsp_iterations' + pre + '.pickle', 'rb') as file:
            iterations = pickle.load(file)
        with open('tsp_rhc' + pre + '.pickle', 'rb') as file:
            rhc_results = pickle.load(file)
        with open('tsp_sa' + pre + '.pickle', 'rb') as file:
            sa_results = pickle.load(file)
        with open('tsp_ga' + pre + '.pickle', 'rb') as file:
            ga_results = pickle.load(file)

        # print iterations
        plt.plot(iterations, rhc_results, label='RHC', color='b')
        plt.plot(iterations, sa_results, label='SA', color='g')
        plt.plot(iterations, ga_results, label='GA', color='y')
        plt.legend()
        if pre == "":
            pre = "start: 100, end: 5000, increment: 100"
        elif pre == "_1000":
            pre = "start: 1000, end: 5000, increment: 1000"
        elif pre == "_5000":
            pre = "start: 1000, end: 50000, increment: 5000"
        plt.title("TSP Iteration Curve with \n" + pre)
        plt.xlabel("Num Iterations")
        plt.ylabel("Optimal Val")
        plt.show()

    with open('fp_iterations.pickle', 'rb') as file:
        iterations = pickle.load(file)
    with open('fp_rhc.pickle', 'rb') as file:
        rhc_results = pickle.load(file)
    with open('fp_sa.pickle', 'rb') as file:
        sa_results = pickle.load(file)
    with open('fp_ga.pickle', 'rb') as file:
        ga_results = pickle.load(file)

    # print iterations
    plt.plot(iterations, rhc_results, label='RHC', color='b')
    plt.plot(iterations, sa_results, label='SA', color='g')
    plt.plot(iterations, ga_results, label='GA', color='y')
    plt.legend()
    plt.title("4-Peaks Iterations Curve")
    plt.xlabel("Num Iterations")
    plt.ylabel("Optimal Val")
    plt.show()

def nn():
    with open('NN_learningCurveSize.pickle', 'rb') as f:
        x = pickle.load(f)
    with open('NN_learningCurveAccuracy.pickle', 'rb') as f:
        data = pickle.load(f)
    for oa in data:
        plt.grid()
        plt.plot(x, data[oa][0], label='Training', color="r")
        plt.plot(x, data[oa][1], label='Testing', color="g")
        plt.xlabel('Training Set Size')
        plt.ylabel('% Accuracy')
        plt.title(oa + ' Learning Curve')
        plt.legend()
        plt.show()

    with open('NN_numIters.pickle', 'rb') as f:
        x = pickle.load(f)
    with open('NN_numIterationsAccuracy.pickle', 'rb') as f:
        data = pickle.load(f)
    for oa in data:
        plt.grid()
        plt.plot(x, data[oa][0], label='Training', color="r")
        plt.plot(x, data[oa][1], label='Testing', color="g")
        plt.xlabel('Num Iterations')
        plt.ylabel('% Accuracy')
        plt.title(oa + ' Iterations Curve')
        plt.legend()
        plt.show()

    plt.show()
