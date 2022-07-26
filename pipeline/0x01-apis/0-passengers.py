#!/usr/bin/env python3
"""Can I join?"""

import requests


def availableShips(passengerCount):
    """availableShips"""
    SWAPI = "https://swapi-api.hbtn.io/api/starships"
    req = requests.get(SWAPI)
    ship_list = req.json()['results']
    available = []
    next = req.json()['next']
    while next:
        for i in ship_list:
            if i['passengers'] == "n/a" or i['passengers'] == "unknown":
                continue
            amount = i['passengers'].replace(",", "")
            if int(amount) > passengerCount:
                available.append(i['name'])
        req = requests.get(next)
        next = req.json()['next']
        ship_list = req.json()['results']
    return available
