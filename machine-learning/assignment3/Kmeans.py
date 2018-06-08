from numpy import *


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, create_Cent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = create_Cent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                print("node ", i, " cluster ", j, " dis ", distJI)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                print("node ", i, "is in cluster", minIndex)
            clusterAssment[i, :] = minIndex, minDist ** 2
        print("LOG:", centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            if len(ptsInClust) > 0:
                centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, disMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))

    # initially create one cluster
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = disMeas(mat(centroid0), dataSet[j, :]) ** 2
    # while the number of cluster is less than k
    while(len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            # perform k mean with k = 2 on given cluster
            ptsInCurrCluster = dataSet[nonzero(
                clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, disMeas)

            # measure total error after k-means
            sseSplit = sum(splitClustAss[:, 1])
            # measure total error before k-means
            sseNotSplit = sum(clusterAssment[nonzero(
                clusterAssment[:, 0].A != i)[0], 1])

            print("SSE Split and not split: ", sseSplit, sseNotSplit)
            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)
                     [0], 0] = bestCentToSplit
        print("the Best cent to split is: ", bestCentToSplit)
        print("the len of bestClusterAss is: ", bestCentToSplit)
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[
            0], :] = bestClustAss
    return mat(centList), clusterAssment
