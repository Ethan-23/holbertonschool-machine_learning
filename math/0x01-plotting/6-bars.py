#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ["Farrah", "Fred", "Felicia"]
bars = [0, 1, 2]

plt.bar(people, fruit[0], color="red", width=0.5, label="apples")
plt.bar(people, fruit[1], color="yellow", width=0.5,
        label="bananas", bottom=fruit[0])
plt.bar(people, fruit[2], color="#ff8000", width=0.5,
        label="oranges", bottom=fruit[1] + fruit[0])
plt.bar(people, fruit[3], color="#ffe5b4", width=0.5,
        label="peaches", bottom=fruit[2] + fruit[1] + fruit[0])

plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.xticks(bars, people)
plt.yticks(np.arange(0, 90, step=10))
plt.legend(loc="upper right")

plt.show()
