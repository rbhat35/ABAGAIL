#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#
export CLASSPATH=../ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot logs image

# # four peaks
# echo "four peaks"
# jython fourpeaks.py

# # count ones
# echo "count ones"
# jython countones.py

# # continuous peaks
# echo "continuous peaks"
# jython continuouspeaks.py

# # knapsack
# echo "Running knapsack"
# jython knapsack.py

# wine test
echo "Running wine test"
jython wine_test.py

# # wine2 test
# echo "Running wine2 test"
# jython wine_test2.py

# abalone test
# echo "Running abalone test"
# jython abalone_test.py

# # traveling salesman
# echo "Running traveling salesman test"
# jython travelingsalesman.py