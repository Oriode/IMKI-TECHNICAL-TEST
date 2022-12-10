# IMKI-TECHNICAL-TEST
IMKI – TECHNICAL TEST

## Introduction
Technical test done for the IMKI company.
The goal was to develop a python script generating a maze using Q-Learning.

## Solution
The solution I founded was to create a maze fully closed with walls everywhere and one after another remove every walls.
After each wall removal, use a Q-Learning IA to find the shortest path and if the IA founds something, stop and return the maze.

## Dependencies
Python >= 3.10.6
Modules : numpy

## Implementation
The maze is implemented using a maxtrix of cells, each cell is an object with a table of neighbors references or None if there is a wall.
In this way each cell can be randomly accessed using (x, y) indexes and the maze can be traveled like a graph starting from any cell to any other.

The Q-Learning IA is a matrix of (nbCells, nbPossiblesActions) where the possibles actions are LEFT, RIGHT, TOP and BOTTOM.
When the action choosed does nothing (moving into a wall), the reward is -1.0 and when moving to a new cell -0.5.
The goal is to find the path with the biggest reward.

## Run
Everything is in one file "qlearning.py" and the lasts lines is the execution for a maze of shape (4, 4).
Simply run the file with a python interpreter and it will works and print the result on the screen.
