from math import sqrt
from PIL import Image, ImageDraw
import numpy as np
import random
import matplotlib.pyplot as plt


def readfile(filename):
    lines = [line for line in file(filename)]

    # First line is the column titles
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        # First column in each row is the rowname
        rownames.append(p[0])
        # The data for this row is the remainder of the row
        data.append([float(x) for x in p[1:]])

    return rownames, colnames, data


def manhattan(v1, v2):
    return np.sum([abs(v1[i]-v2[i]) for i in range(len(v1))])


def tanamoto(v1, v2):
    c1, c2, shr = 0, 0, 0

    for i in range(len(v1)):
        if v1[i] != 0:
            c1 += 1
        if v2[i] != 0:
            c2 += 1
        if v1[i] != 0 and v2[i] != 0:
            shr += 1

    return 1.0-(float(shr)/(c1+c2-shr))


def pythagorean(v1, v2):
    return np.sqrt(np.sum([pow(v1[i]-v2[i], 2) for i in range(len(v1))]))


def pearson(v1, v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)

    # Sums of the squares
    sum1_sq = sum([pow(v, 2) for v in v1])
    sum2_sq = sum([pow(v, 2) for v in v2])

    p_sum = sum([v1[i]*v2[i] for i in range(len(v1))])

    num = p_sum - (sum1*sum2/len(v1))
    den = sqrt((sum1_sq - pow(sum1, 2)/len(v1)) * (sum2_sq - pow(sum2, 2)/len(v2)))
    if den == 0:
        return 0

    return 1.0-num/den


def pearson_alpha(v1, v2):
    x, y = v1, v2
    multi = [x[i] * y[i] for i in range(len(x))]
    cov = np.average(multi) - np.average(x)*np.average(y)
    den = np.std(x)*np.std(y)

    if den == 0:
        return 0

    return 1.0-cov/den


class BiCluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance


def hcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1

    # Clusters are initially just the rows
    clust = [BiCluster(rows[i], id=i) for i in range(len(rows))]

    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                # distances is the cache of distance calculations
                if(clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)

                d = distances[(clust[i].id, clust[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i, j)

        # calculate the average of the two clusters
        mergevec = [(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]

        # create the new cluster
        new_cluster = BiCluster(mergevec, left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance=closest,
                                id=currentclustid)

        # cluster ids that weren't in the original set are negative
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(new_cluster)

    return clust[0]


def printclust(clust, labels=None, n=0):
    # indent to make a hierarchy layout
    for i in range(n):
        print ' ',
    if clust.id < 0:
        # negative id means that this is branch
        print '-'
    else:
        # positive id means that is an endpoint
        if labels is None:
            print clust.id
        else:
            print labels[clust.id]

    if clust.left is not None:
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right is not None:
        printclust(clust.right, labels=labels, n=n+1)


def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left is None and clust.right is None:
        return 1
    # Otherwise the height is the same of the heights of each branch
    return getheight(clust.left)+getheight(clust.right)


def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left is None and clust.right is None:
        return 0
    # The distance of a branch is the greater of its two sides plus its own distance
    return max(getdepth(clust.left), getdepth(clust.right))+clust.distance


def drawdendrogram(clust, labels, jpeg='clusters.jpg'):

    h = getheight(clust)*20
    w = 1200
    depth = getdepth(clust)

    scaling = float(w-150)/depth

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.line((0, h/2, 10, h/2), fill=(255, 0, 0))

    drawnode(draw, clust, 10, (h/2), scaling, labels)
    img.save(jpeg, 'JPEG')


def drawnode(draw, clust, x, y, scaling, labels):
    if clust.id < 0:
        h1 = getheight(clust.left)*20
        h2 = getheight(clust.right)*20
        top = y-(h1+h2)/2
        bottom = y+(h1+h2)/2

        ll = clust.distance*scaling
        # Vertical line from this cluster to children
        draw.line((x, top+h1/2, x, bottom-h2/2), fill=(255, 0, 0))

        # Horizontal line to left item
        draw.line((x, top+h1/2, x+ll, top+h1/2), fill=(255, 0, 0))

        # Horizontal line to right item
        draw.line((x, bottom-h2/2, x+ll, bottom-h2/2), fill=(255, 0, 0))

        drawnode(draw, clust.left, x+ll, top+h1/2, scaling, labels)
        drawnode(draw, clust.right, x+ll, bottom-h2/2, scaling, labels)
    else:
        draw.text((x+5, y-7), labels[clust.id], (0, 0, 0))


def rotatematrix(data):
    newdata = []
    for i in range(len(data[0])):
        newrow = [data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata


def kcluster(rows, distance=pearson_alpha, k=4, iter=100):
    # Determine the minimum and maximum values for each point
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows])) for i in range(len(rows[0]))]

    # Create k randomly placed centroids
    clusters = [[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]

    lastmatches = []
    for t in range(iter):
        print 'Iteration %d' % t

        bestmatches = [[] for i in range(k)]

        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)

        # If the results are the same as last time, this is complete
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        # Move the centroids to the average of their members
        for i in range(k):
            avgs = [0.0]*len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs

    return bestmatches


def scaledown(data, distance=tanamoto, rate=0.01):
    n = len(data)

    # The real distances between every pair of items
    realdist = [[distance(data[i], data[j]) for j in range(n)] for i in range(0, n)]

    max = np.max([np.max(i) for i in realdist])
    min = np.min([np.min(i) for i in realdist])
    realdist = [[(realdist[i][j]-min)/(max-min) for j in range(n)] for i in range(n)]

    # Randomly initialize of the starting points of the locations in 2D
    loc = [[random.random(), random.random()] for i in range(n)]

    fakedist = [[0.0 for j in range(n)] for i in range(n)]

    lasterror = None
    for m in range(0, 1000):

        for i in range(n):
            for j in range(n):
                fakedist[i][j] = sqrt(sum([pow(loc[i][x]-loc[j][x], 2) for x in range(len(loc[i]))]))

        # Move points
        grad = [[0.0, 0.0] for i in range(n)]

        totalerror = 0
        for k in range(n):
            for j in range(n):
                if j == k:
                    continue
                # The error is percent difference between the distances
                errorterm = (fakedist[j][k] - realdist[j][k])/realdist[j][k]
                sign = 0
                if fakedist[j][k] > realdist[j][k]:
                    sign = 1
                elif fakedist[j][k] < realdist[j][k]:
                    sign = -1
                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                grad[k][0] += ((loc[k][0] - loc[j][0])/fakedist[j][k])*errorterm
                grad[k][1] += ((loc[k][1] - loc[j][1])/fakedist[j][k])*errorterm

                # Keep track of the total error
                totalerror += abs(errorterm)
        print totalerror

        # If the answer got worse by moving the points, we are done
        if lasterror and lasterror < totalerror:
            break
        lasterror = totalerror

        # Move each of the points by the learning rate times the gradient
        for k in range(n):
            loc[k][0] -= rate*grad[k][0]
            loc[k][1] -= rate*grad[k][1]
        # if m % 10 == 0:
        #     scatter(loc)

    return loc


def draw2d(data, labels, jpeg='mds2d.jpg'):
    img = Image.new('RGB', (2000, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(len(data)):
        x = (data[i][0] + 0.5)*1000
        y = (data[i][1] + 0.5)*1000
        draw.text((x, y), labels[i], (0, 0, 0))
    img.save(jpeg, 'JPEG')


def scatter(loc):
    m = len(loc)/2
    x1 = [i[0] for i in loc[0:m]]
    y1 = [i[1] for i in loc[0:m]]
    x2 = [i[0] for i in loc[m:]]
    y2 = [i[1] for i in loc[m:]]

    plt.figure()
    plt.scatter(x1, y1, c='r')
    plt.scatter(x2, y2, c='b')
    plt.show()


def main():
    # blognames, words, data = readfile('blogdata.txt')
    # clust = hcluster(data)
    # drawdendrogram(clust, labels=blognames, jpeg='blogclust.jpg')
    # rdata = rotatematrix(data)
    # wordclust = hcluster(rdata)
    # drawdendrogram(wordclust, labels=words, jpeg='wordclust.jpg')
    wants, people, data = readfile('zebo.txt')
    clust = scaledown(data, distance=manhattan, rate=0.01)
    draw2d(clust, wants, jpeg='wants2d.jpg')

    # blognames, words, data = readfile('blogdata.txt')
    # clust = scaledown(data, distance=pythagorean, rate=0.002)
    # draw2d(clust, blognames, jpeg='blogs2d.jpg')


if __name__ == '__main__':
    main()
