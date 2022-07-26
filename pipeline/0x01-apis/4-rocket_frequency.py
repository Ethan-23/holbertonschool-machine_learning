#!/usr/bin/env python3
"""Rate me is you can!"""

import requests
import sys


if __name__ == '__main__':
    url = "https://api.spacexdata.com/"
    count = dict()
    req = requests.get(url + "v4/launches/").json()

    for i in req:
        rocket = requests.get(url + "v4/rockets/" + i['rocket']).json()
        if rocket['name'] in count.keys():
            count[rocket['name']] += 1
        else:
            count[rocket['name']] = 1

    sorted_count = sorted(count.items(),
                          key=lambda item: (item[1], item[0]),
                          reverse=True)

    for k, v in sorted_count:
        print("{}: {}".format(k, v))
