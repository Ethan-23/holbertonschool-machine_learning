#!/usr/bin/env python3
"""Rate me is you can!"""

from datetime import datetime
import requests
import sys


if __name__ == '__main__':
    req = requests.get(sys.argv[1])
    if req.status_code == 404:
        print("Not found")
    elif req.status_code == 403:
        time = int(
            (
                datetime.fromtimestamp(
                    int(req.headers['X-RateLimit-Reset']))
                - datetime.now()
            ).total_seconds() / 60
        )
        print('Reset in {} min'.format(time))
    elif req.ok:
        print(req.json()['location'])
