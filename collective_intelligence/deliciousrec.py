from pydelicious import get_popular, get_urlposts, get_userposts
from time import time


def initialize_user_dict(tag, count=5):
    user_dict = {}
    # get the top count popular posts
    cn = 0
    for p1 in get_popular(tag=tag)[0:count]:
        # find all users who posted this
        cn += 1
        print cn
        print p1['url']
        for p2 in get_urlposts(p1['url']):
            user = p2['user']
            user_dict[user] = {}
    print user_dict
    return user_dict


def fill_items(user_dict):
    all_items = {}
    for user in user_dict:
        for i in range(3):
            try:
                posts = get_userposts(user)
                break
            except:
                print "Failed user "+user+", retrying"
                time.sleep(4)
        for post in posts:
            url = post['url']
            user_dict[user][url] = 1.0
            all_items[url] = 1

    # Fill in missing items with 0
    for ratings in user_dict.values():
        for item in all_items:
            if item not in ratings:
                ratings[item] = 0.0


def main():
    delusers = initialize_user_dict('girl', 1)
    fill_items(delusers)

if __name__ == '__main__':
    main()