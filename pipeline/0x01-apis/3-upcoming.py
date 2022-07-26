#!/usr/bin/env python3
"""Rate me is you can!"""

import requests
import sys


if __name__ == '__main__':
    url = "https://api.spacexdata.com/"

    req = requests.get(url + "v5/launches/latest")

    json = req.json()

    launch_name = json['name']
    date = json['date_local']

    req = requests.get(url + "v4/rockets/" + json['rocket'])

    rocket_name = req.json()['name']

    req = requests.get(url + "v4/launchpads/" + json['launchpad'])

    launchpad_name = req.json()['name']
    launchpad_locality = req.json()['locality']

    print('{} ({}) {} - {} ({})'.format(launch_name, date, rocket_name,
                                        launchpad_name, launchpad_locality))
