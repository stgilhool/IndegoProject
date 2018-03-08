#script that makes transition matrix and propagates trips
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os.path
import pdb
import code
import time
import networkx as nx
from show_route_info import get_route_index
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pylab

plt.rcParams['figure.figsize'] = (20.0, 20.0)

filename = "indego-trips-2017-q3.csv"
#file_dir = "C:\\Users\Eric\\Desktop\\job_search\\data_science\\project\\data"
file_dir = filename

#navigate to data file
#os.chdir(file_dir)

#initialize arrays
trip_id = []
start_station = []
end_station = []


#open CSV file; loop through the file and import data into the selected arrays
with open(filename) as testCSV:
    test1 = csv.reader(testCSV)
    ii = 0
    for row in test1:
        if ii > 0:

            trip_id.append(row[0])
            start_station.append(row[4])
            end_station.append(row[7])

        ii = ii + 1

#convert station lists from strings into list of integers
start_stationInt = list(map(int, start_station))
end_stationInt = list(map(int, end_station))

# Convert lists into np arrays, and then to int type
# find the number of unique entries in this array

startStat = np.array(start_station)
startStatInt = startStat.astype(int)

endStat = np.array(end_station)
endStatInt = endStat.astype(int)

# Get vector of unique station ID's
startStatUn = np.unique(startStatInt)
endStatUn = np.unique(endStatInt)

# Make sure that startStatUn and endStatUn contain all the same stations
stationCheck = np.array_equal(startStatUn, endStatUn)
if not stationCheck:
    sys.exit("ERROR: start and end station vectors contain different stations")


# vector of route identifiers (string concat of start + end)
routeIDtot = [startStat[i]+endStat[i] for i in range(len(startStat))]

#turn this vector into a numpy array and then into an integer array
routeID = np.array(routeIDtot)
# print(type(routeID))
routeID = routeID.astype(int)

#code.interact(local=locals())

primeQ = np.zeros([len(startStatUn), len(startStatUn)])

for rowIdx in range(len(startStat)):
    # Get Q row index
    startStatRow = startStatInt[rowIdx]
    rowIdxQ = np.where(startStatUn == startStatRow)

    endStatRow = endStatInt[rowIdx]
    colIdxQ = np.where(startStatUn == endStatRow)

    primeQ[rowIdxQ, colIdxQ] = primeQ[rowIdxQ, colIdxQ] + 1

matrixQ = primeQ/(primeQ.sum(axis=1, keepdims=True))

tol = 0.01
matrixQ[matrixQ < tol ] = 0

# renormalize the Q matrix
Qnorm = np.abs(matrixQ).sum(axis=1)
matrixQ = matrixQ.astype(np.float) / Qnorm[:,None]

# let's try to make a graph of the network using our adjacency/transition matrix


# code.interact(local=locals())

# array to hold list of graph edges.
edgeTot = []

for sIdx in range(len(matrixQ)):
    edgeTot.append(np.nonzero(matrixQ[sIdx])[0])

edgeTot = np.array(edgeTot)
Qedge=[]
QedgeI = []
QedgeJ = []
edgeW = []

for iNode in range(len(edgeTot)):
   for jNode in edgeTot[iNode]:
       Qedge.append((iNode,jNode))
       QedgeI.append(iNode)
       QedgeJ.append(jNode)
       edgeW.append(matrixQ[iNode][jNode])

weightList = []

for weightVal in range(len(edgeW)):
    weightList.append(dict(weight = edgeW[weightVal]))


QedgeTot = list(zip(QedgeI,QedgeJ,weightList))

QnodesTot = np.arange(0,124)

G = nx.DiGraph()

G.add_nodes_from(QnodesTot)
G.add_edges_from(QedgeTot)

pos = nx.spring_layout(G,k = 0.8, iterations = 100)

edgeMin = min(edgeW)
edgeMax = max(edgeW)

edgeHeat = edgeCm = plt.get_cmap('gist_heat')
cNormEdge = colors.LogNorm(vmin = edgeMin, vmax = edgeMax)
scalarMapEdge = cmx.ScalarMappable(norm = cNormEdge, cmap = edgeHeat)
edgeColSet = []
for edge in range(len(edgeW)):
    edgeColVal = scalarMapEdge.to_rgba(edgeW[edge])
    edgeColSet.append(edgeColVal)


Gdeg = nx.degree(G)
degScale = []
for deg in range(len(Gdeg)):
    degScale.append(Gdeg[deg])


nodeLabels = {node:node for node in QnodesTot}

# nx.draw(G,pos,edge_color = edgeColSet,node_size = 1000, node_color=degScale,cmap = plt.cm.coolwarm)
# nx.draw_networkx_labels(G,pos,labels = nodeLabels, font_size = 12)
# nx.draw_networkx_edges(G,pos,alpha = 0.01)

# plt.show()

######uniform initial distribution test#######

# generate uniform probability vector
uniProb = 1/len(QnodesTot)
initDist = np.ones(124)*uniProb

# set tolerance condition for breaking the loop
loopTol = 1e-3

#initialize arrays and feed them the initial state
simDist = []
chiDist = []
nodeSize = []

simDist.append(initDist)

nodeSize.append(1e5*initDist)

chiRes = np.zeros(124)
chiDist.append(chiRes)

# propogate the bike density forward
nSteps = 10

for simIter in range(1,1+nSteps):
    result = np.matmul(simDist[simIter-1], matrixQ)
    simDist.append(result)

    nodeSize.append(1e5*simDist[simIter])

    chiRes = abs(simDist[simIter] - simDist[simIter-1])
    chiDist.append(chiRes)

    print(max(chiDist[simIter]))

    if max(chiDist[simIter]) < loopTol:
        break

iterMax = simIter

# code.interact(local=locals())

# generate plots and save them
graph = nx.Graph()
nodeSizeIdx = 0
pylab.ion()

def make_bike_graph():
    global nodeSize
    global nodeSizeIdx
    nodeSizeIdx += 1
    nx.draw(G,pos,edge_color = edgeColSet,node_size = nodeSize[nodeSizeIdx], node_color=degScale,cmap = plt.cm.coolwarm)
    nx.draw_networkx_labels(G,pos,labels = nodeLabels, font_size = 12)
    nx.draw_networkx_edges(G,pos,alpha = 0.01)

pylab.show()

for simIter in range(iterMax):
    if simIter == 0:
        nx.draw(G,pos,edge_color = edgeColSet,node_size = nodeSize[0], node_color=degScale,cmap = plt.cm.coolwarm)
        nx.draw_networkx_labels(G,pos,labels = nodeLabels, font_size = 12)
        nx.draw_networkx_edges(G,pos,alpha = 0.01)

        pylab.draw()
        plt.pause(1)
        plt.savefig("uniform" + str(simIter) + ".png")
        plt.clf()

    elif simIter > 0:
        fig = make_bike_graph()
        pylab.draw()
        plt.pause(1)
        plt.savefig("uniform" + str(simIter) + ".png")
        plt.clf()

# fig = plt.figure()
# graphSet, = nx.DiGraph()

# def init():
#     graphSet, = nx.Digraph()
#     return graphSet,
#
# #animation function. This is called sequentially
# def animate(i):
#     global nodeSize
#     global nodeSizeIdx
#     global pos
#     global degScale
#     nx.draw(G,pos,edge_color = edgeColSet,node_size = nodeSize[nodeSizeIdx], node_color=degScale,cmap = plt.cm.coolwarm)
#     nx.draw_networkx_labels(G,pos,labels = nodeLabels, font_size = 12)
#     nx.draw_networkx_edges(G,pos,alpha = 0.01)

#
#
code.interact(local=locals())

# bikevec = np.ones(len(startStatUn))
#
# trip = np.matmul(bikevec, matrixQ)
#
# trip2 = np.matmul(trip, matrixQ)
#
# fig = plt.figure(figsize=(12,9))
# plt.ion()
# plt.show()
#
# for tripnum in range(0,20):
#
#     if tripnum > 0:
#         result = np.matmul(bikevec, matrixQ)
#         bikevec = result
#
#     elif tripnum == 0:
#         result = bikevec
#
#     trip = fig.add_subplot(111)
#     titleString = 'Trip '+str(tripnum)
#     trip.set_title(titleString)
#     trip.set_autoscaley_on(False)
#     trip.set_ylim([0,5])
#     trip.plot(result, linestyle='None', marker='.')
#
#     plt.draw()
#
#     if tripnum == 0:
#         plt.pause(3)
#     else:
#         plt.pause(1)
#
#     fig.clf()
#
#
# code.interact(local=locals())
