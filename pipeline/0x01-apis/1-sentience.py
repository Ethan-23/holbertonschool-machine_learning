#!/usr/bin/env python3
"""Can I join?"""

import requests


def sentientPlanets():
    """availableShips"""
    SWAPI = "https://swapi-api.hbtn.io/api/people"
    req = requests.get(SWAPI)
    people_list = req.json()['results']
    available = []
    next = req.json()['next']
    while next:
        for i in people_list:
            home_planet = i['homeworld']
            hp = requests.get(home_planet)
            if hp.json()['name'] in available:
                continue
            for k in i['species']:
                spe = requests.get(k)
                if spe.json()['designation'] == "sentient":
                    available.append(hp.json()['name'])
                    break
        req = requests.get(next)
        next = req.json()['next']
        people_list = req.json()['results']
    return available
