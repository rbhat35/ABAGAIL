import matplotlib.pyplot as plt
import pickle

with open('tsp_iterations_5000.pickle', 'rb') as file:
    iterations = pickle.load(file)
with open('tsp_rhc_5000.pickle', 'rb') as file:
    rhc_results = pickle.load(file)
with open('tsp_sa_5000.pickle', 'rb') as file:
    sa_results = pickle.load(file)
with open('tsp_ga_5000.pickle', 'rb') as file:
    ga_results = pickle.load(file)

# print iterations
plt.plot(iterations, rhc_results, label='RHC', color='b')
plt.plot(iterations, sa_results, label='SA', color='g')
plt.plot(iterations, ga_results, label='GA', color='y')
plt.legend()
plt.xlabel("Num Iterations")
plt.ylabel("Optimal Val")


plt.show()


