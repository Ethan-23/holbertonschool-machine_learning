#!/usr/bin/env python3
"""Create the loop"""

exit_list = ["exit", "quit", "goodbye", "bye"]

while True:
    question = input("Q: ")
    if question.lower() in exit_list:
        break
    print("A: ")
print("A: Goodbye")
