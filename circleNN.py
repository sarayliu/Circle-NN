# Sara Liu 5/7/19

import sys
import random
import math
import time


def ff(weights, func, input):
    prevNodeCt = len(input)
    newInput = input[:]
    ffList = [input]
    for layer in range(len(weights)):
        layerWeights = [weight for weight in weights[layer]]
        if layer < len(weights) - 1:
            nodeWeights = [layerWeights[idx:idx + prevNodeCt] for idx in range(0, len(layerWeights), prevNodeCt)]
            nodeValues = []
            for node in nodeWeights:
                nodeValues.append(transferFunction(func, dotProduct(newInput, node)))
            newInput = nodeValues[:]
            prevNodeCt = len(newInput)
            ffList.append(newInput)
        else:
            output = []
            nodeCt = 0
            for weight in layerWeights:
                output.append(newInput[nodeCt] * weight)
                if nodeCt < len(newInput) - 1:
                    nodeCt += 1
            ffList.append(output)
    return ffList


def bp(ffList, weights, target):
    bpList = ffList[:]
    bpList[-1] = [target - ffList[-1][0]]
    for layer in range(len(bpList) - 2, 0, -1):
        # bpList[layer] = [weights[layer][node] * bpList[layer + 1][0] * ffList[layer][node] * (1 - ffList[layer][node]) for node in range(len(ffList[layer]))]
        bpList[layer] = []
        for node in range(len(ffList[layer])):
            bpList[layer].append(dotProduct(weights[layer][node::len(ffList[layer])], bpList[layer + 1]) * ffList[layer][node] * (1 - ffList[layer][node]))
    return bpList


def negGradient(ffList, bpList, numLayers):
    gradList = [[bpList[i + 1][idx // len(ffList[i])] * ffList[i][idx % len(ffList[i])] for idx in range(numLayers[i] * numLayers[i + 1])] for i in range(len(numLayers) - 1)]
    return gradList


def newWeights(weights, gradList):
    for weightLayer in range(len(weights)):
        for weight in range(len(weights[weightLayer])):
            weights[weightLayer][weight] = gradList[weightLayer][weight] * 0.01 + weights[weightLayer][weight]
    return weights


def dotProduct(vector1, vector2):
    if len(vector1) != len(vector2):
        return False
    dp = 0
    for idx in range(len(vector1)):
        dp += vector1[idx] * vector2[idx]
    return dp


def transferFunction(func, x):
    if func == 'T1':
        return x
    if func == 'T2':
        if x > 0:
            return x
        else:
            return 0
    if func == 'T3':
        return 1/(1 + math.exp(-x))
    if func == 'T4':
        return -1 + 2/(1 + math.exp(-x))
    else:
        return 'Function not valid'


t1 = time.time()
inputExp = sys.argv[1]
inputs, targets = [], []
for iteration in range(2000):
    x, y = random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)
    inputs.append([x, y, 1])
    if eval(inputExp):
        targets.append(1)
    else:
        targets.append(0)
layerCts = [len(inputs[0]), 18, 6, 2, 1, 1]
weightList = [[random.uniform(-2.0, 2.0) for num in range(layerCts[i] * layerCts[i + 1])] for i in range(len(layerCts) - 1)]
print('Layer cts: ', layerCts)
loopCt = 0
incorrectCt = 0
while True:
    testCase = loopCt % 2000
    feedForward = ff(weightList, 'T3', inputs[testCase])
    # print(feedForward)
    if targets[testCase] != round(feedForward[-1][0]):
        incorrectCt += 1
    if loopCt % 2000 == 1999:
        if incorrectCt > 10:
            inputs, targets = [], []
            for iteration in range(2000):
                x, y = random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)
                inputs.append([x, y, 1])
                if eval(inputExp):
                    targets.append(1)
                else:
                    targets.append(0)
            loopCt = 0
            incorrectCt = 0
            # for wList in weightList:
            #     print(wList)
            continue
        else:
            break
    backPropagation = bp(feedForward, weightList, targets[testCase])
    # print(backPropagation)
    gradient = negGradient(feedForward, backPropagation, layerCts)
    # print(gradient)
    weightList = newWeights(weightList, gradient)
    # for wList in weightList:
    #     print(wList)
    loopCt += 1
print('Weights')
for wList in weightList:
    print(wList)
t2 = time.time()
# print('Time: ', t2 - t1, 's')
