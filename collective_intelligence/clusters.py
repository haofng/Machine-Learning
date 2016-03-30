from math import sqrt
from PIL import Image, ImageDraw
import numpy as np


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


def main():
    blognames, word, data = readfile('blogdata.txt')
    clust = hcluster(data)


if __name__ == '__main__':
    main()
