from math import sqrt
import numpy as np
# A dictionary of movie critics and their ratings of a small set of movies
critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                         'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},
           'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5,
                            'Superman Returns': 5.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 3.5},
           'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, 'Superman Returns': 3.5,
                                'The Night Listener': 4.0},
           'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5,
                            'Superman Returns': 4.0, 'You, Me and Dupree': 2.5},
           'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0,
                            'Superman Returns': 3.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 2.0},
           'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'The Night Listener': 3.0,
                             'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
           'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}
           }


# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs, person1, person2):
    # Get the list of shared_items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    # if they have no rating in common, return 0
    if len(si) == 0:
        return 0

    # Add up the squares of all the differences
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item], 2) for item in prefs[person1] if item in prefs[person2]])

    return 1/(1+sum_of_squares)


# Returns the Pearson correlation coefficient for p1 and p2
# r = cov(x,y)/(std(x)*std(y))
# cov(x,y) = np.average(a*b)-np.average(a)*np.average(b)
def sim_pearson(prefs, p1, p2):
    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    n = len(si)

    if n == 0:
        return 0

    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])

    sum1_sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2_sq = sum([pow(prefs[p2][it], 2) for it in si])

    p_sum = sum([prefs[p1][it]*prefs[p2][it] for it in si])

    num = p_sum - (sum1*sum2/n)
    den = sqrt((sum1_sq-pow(sum1, 2)/n) * (sum2_sq-pow(sum2, 2)/n))
    if den == 0:
        return 0

    r = num/den
    return r


# Returns the Pearson correlation coefficient for p1 and p2
# r = cov(x,y)/(std(x)*std(y))
# cov(x,y) = np.average(a*b)-np.average(a)*np.average(b)
def sim_pearson_alpha(prefs, p1, p2):
    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    n = len(si)

    if n == 0:
        return 0

    x = [prefs[p1][it] for it in si]
    y = [prefs[p2][it] for it in si]
    multi = [x[i] * y[i] for i in range(len(x))]
    cov = np.average(multi) - np.average(x)*np.average(y)

    return cov/(np.std(x)*np.std(y))


# Return the best matches for person from the prefs dictionary.
# Number of results and similarity function are optional params.
def top_matches(prefs, person, n=5, similarity=sim_pearson_alpha):
    scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]

    # Sort the list so the hightest scores appear at the top
    scores.sort()
    scores.reverse()
    return scores[0:n]


# Gets recommendations for a person by using a weighted average of every other user's rankings
def get_recommendations(prefs, person, similarity=sim_pearson_alpha):
    totals = {}
    sim_sums = {}
    for other in prefs:
        # dont't compare me to myself
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in prefs[other]:
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item] == 0:
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item]*sim
                # Sum of similarities
                sim_sums.setdefault(item, 0)
                sim_sums[item] += sim
    # Create the normalized list
    rankings = [(total/sim_sums[item], item) for item, total in totals.items()]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


def transform_prefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})

            # Flip item and person
            result[item][person] = prefs[person][item]
    return result


def main():
    persons = [key for key in critics.iterkeys()]
    matric_pearson = [[sim_pearson_alpha(critics, p1, p2) for p1 in persons] for p2 in persons]
    matric_distance = [[sim_distance(critics, p1, p2) for p1 in persons] for p2 in persons]
    print 'distance:'
    print_matric(matric_distance)
    print 'pearson:'
    print_matric(matric_pearson)


def print_matric(matric):
    for i in range(len(matric)):
        row = matric[i]
        for j in range(len(row)):
            print '%.2f' % matric[i][j],
        print ''


if __name__ == '__main__':
    main()