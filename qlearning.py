from __future__ import annotations

import numpy
from enum import Enum
import random
import sys

class Log:

	DEBUG = True

	@staticmethod
	def displayLog(msg : str) -> None:
		if Log.DEBUG is True:
			print(msg, file=sys.stdout, flush=True)

	def displayError(msg : str) -> None:
		print(msg, file=sys.stderr, flush=True)

"""
	@brief Enum representing the type of a maze cell.
"""
class CellType(Enum):
	Empty = 0
	Entry = 1
	Exit = 2
	Treasure = 3

"""
	@brief Enum representing the Action possible when resolving path using Q-Learning.
"""
class Action(Enum):
	Left = 0
	Right = 1
	Top = 2
	Bottom = 3

"""
	@brief Cell of the maze, connected to others by 4 others ways (top, bottom, left & right). 
"""
class Cell:
	
	neighbors : numpy.ndarray
	
	type : CellType
	cellNum : int
	
	def __init__(self, cellNum : int) -> None:
		self.neighbors = numpy.empty((len(Action)), dtype=object)
		self.neighbors[:] = None
		self.type = CellType.Empty
		self.cellNum = cellNum
	
	def __str__(self) -> str:
		return f'[#{self.cellNum}: {self.type.name}]'

	def getCellNum(self) -> int:
		return self.cellNum

	def getNeighbour(self, action : Action) -> Cell:
		return self.neighbors[action.value]
	
	def setNeighbour(self, action : Action, cell : Cell) -> Cell:
		self.neighbors[action.value] = cell
	
	def setCellNum(self, cellNum : int) ->  None:
		self.cellNum = cellNum
	
	def getType(self) -> CellType:
		return self.type
	
	def setType(self, cellType : CellType) -> None:
		self.type = cellType

"""
	@brief Maze with an entry, a treasure and an exit.
"""
class Maze:

	cellMat : numpy.ndarray		# 2D Matrix of Cells.
	entryCell : Cell
	exitCell : Cell
	treasureCell : Cell

	def __init__(self, shape : numpy.shape) -> None:
		self.cellMat = numpy.empty(shape, dtype=object)
		for y in range(0, self.cellMat.shape[1]):
			for x in range(0, self.cellMat.shape[0]):
				self.cellMat[x, y] = Cell(y * self.cellMat.shape[1] + x)
		self.reset()
	
	def __str__(self) -> str:

		outputStr : str = ''

		# Start to print the top walls.
		outputStr += '█'
		for x in range(0, self.cellMat.shape[0]):
			outputStr += '██'
		outputStr += '\n'
			
		# Next print each cell.
		for y in range(0, self.cellMat.shape[1]):
			outputStr += '█'
			for x in range(0, self.cellMat.shape[0]):
				cell : Cell = self.cellMat[x, y]
				if cell.getType().value == CellType.Empty.value:
					outputStr += ' '
				elif cell.getType().value == CellType.Entry.value:
					outputStr += 'S'
				elif cell.getType().value == CellType.Exit.value:
					outputStr += 'E'
				elif cell.getType().value == CellType.Treasure.value:
					outputStr += 'X'
				if cell.getNeighbour(Action.Right) is None:
					outputStr += '█'
				else:
					outputStr += ' '
			outputStr += '\n'
			outputStr += '█'
			for x in range(0, self.cellMat.shape[0]):
				cell : Cell = self.cellMat[x, y]
				if cell.getNeighbour(Action.Bottom) is None:
					outputStr += '██'
				else:
					outputStr += ' █'
			outputStr += '\n'
		
		return outputStr

	def getCellXY(self, x : int, y : int) -> Cell:
		if x < 0 or x >= self.cellMat.shape[0]:
			return None
		if y < 0 or y >= self.cellMat.shape[1]:
			return None
		return self.cellMat[x, y]

	def getCell(self, cellNum : int) -> Cell:
		y : int = cellNum // self.cellMat.shape[0]
		x : int = cellNum % self.cellMat.shape[0]
		return self.getCellXY(x, y)
	
	def getNbCells(self) -> int:
		return self.cellMat.size

	"""
		Reset the maze with walls everywhere and only empty cells.
	"""
	def reset(self) -> None:
		for y in range(0, self.cellMat.shape[1]):
			for x in range(0, self.cellMat.shape[0]):
				cell : Cell = self.cellMat[x, y]

				cell.setNeighbour(Action.Left, None)
				cell.setNeighbour(Action.Right, None)
				cell.setNeighbour(Action.Top, None)
				cell.setNeighbour(Action.Bottom, None)
				cell.setType(CellType.Empty)
		
		self.entryCell = None
		self.exitCell = None
		self.treasureCell = None
	
	def setRandomlySpecialsCells(self) -> bool:
		self.entryCell = self.setRandomlySpecialsCell(CellType.Entry)
		self.exitCell = self.setRandomlySpecialsCell(CellType.Exit)
		self.treasureCell = self.setRandomlySpecialsCell(CellType.Treasure)

		return self.entryCell is not None and self.exitCell is not None and self.treasureCell is not None
		
	def setRandomlySpecialsCell(self, cellType : CellType) -> Cell:
		for i in range(0, 1000):
			x : int = random.randint(0, self.cellMat.shape[0] - 1)
			y : int = random.randint(0, self.cellMat.shape[1] - 1)

			cell : Cell = self.getCellXY(x, y)

			if cell.getType().value != CellType.Empty.value:
				continue
			cell.setType(cellType)
			return cell
		return None

	def removeWallRandomly(self) -> bool:
		for i in range(0, 1000):
			x : int = random.randint(0, self.cellMat.shape[0] - 1)
			y : int = random.randint(0, self.cellMat.shape[1] - 1)

			cell : Cell = self.getCellXY(x, y)
			
			wallToRemove : int = random.randint(0, 3)

			if wallToRemove == 0:
				if x > 0:
					if cell.getNeighbour(Action.Left) is None:
						otherCell : Cell = self.getCellXY(x - 1, y)
						cell.setNeighbour(Action.Left, otherCell)
						otherCell.setNeighbour(Action.Right, cell)
						return True
			elif wallToRemove == 1:
				if y > 0:
					if cell.getNeighbour(Action.Top) is None:
						otherCell : Cell = self.getCellXY(x, y - 1)
						cell.setNeighbour(Action.Top, otherCell)
						otherCell.setNeighbour(Action.Bottom, cell)
						return True
			elif wallToRemove == 2:
				if x < self.cellMat.shape[0] - 1:
					if cell.getNeighbour(Action.Right) is None:
						otherCell : Cell = self.getCellXY(x + 1, y)
						cell.setNeighbour(Action.Right, otherCell)
						otherCell.setNeighbour(Action.Left, cell)
						return True
			elif wallToRemove == 3:
				if y < self.cellMat.shape[1] - 1:
					if cell.getNeighbour(Action.Bottom) is None:
						otherCell : Cell = self.getCellXY(x, y + 1)
						cell.setNeighbour(Action.Bottom, otherCell)
						otherCell.setNeighbour(Action.Top, cell)
						return True

		return False
	
	def getEntryCell(self) -> Cell:
		return self.entryCell
	
	def getExitCell(self) -> Cell:
		return self.exitCell
	
	def getTreasureCell(self) -> Cell:
		return self.treasureCell

			

"""
	@brief Class used to find a path from an entry to an exit using Q-Learning.
"""
class PathFinder:

	maze : Maze				# Maze used to compute the path on.
	Q : numpy.ndarray		# Matrix of hope computed with Q-Learning.

	def __init__(self, maze : Maze) -> None:
		self.maze = maze
		self.Q = numpy.zeros((self.maze.getNbCells(), len(Action)))
		self.reset()

	def reset(self) -> None:
		self.Q.fill(0.0)
	
	def takeAction(self, cell : Cell, epsilon : float) -> Action:
		if random.uniform(0.0, 1.0) < epsilon:
			# Take a random descision
			selectedAction : Action = Action(random.randint(0, len(Action) - 1))
			return selectedAction
		else:
			# Take a greedy action
			return Action(numpy.argmax(self.Q[cell.getCellNum()]))
	
	"""
		@brief	Update Q using Q-Learning to find a path from the entryCell to exitCell.
		@param entryCell	begin Cell.
		@param exitCell		Cell to be reached.
		@param nbLoops		Number of iterations to be done.
		@param maxTries		Number of Tries for each loop before canceling.
		@param epsilon		Value between [0.0;1.0] represent the randomness of the search.
		@param learningRate	Speed used to modify Q.
		@param gamma		Offset used to add more value to current values intead of old ones.
	"""
	def learn(self, entryCell : Cell, exitCell : Cell, nbLoops : int = 100, maxTries : int = 10000, epsilon : float = 0.1, learningRate : float = 0.01, gamma : float = 0.9) -> bool:
		
		founded : bool = False

		for i in range(0, nbLoops):
			currentCell : Cell = entryCell
			for i in range(0, maxTries):
				if currentCell == exitCell:
					# Log.displayLog(f'Action : {actionP0.name} ({currentCell}).')
					founded = True
					break
				
				# Get the next Action.
				actionP0 : Action = self.takeAction(currentCell, epsilon)
				cellP1 : Cell = currentCell.getNeighbour(actionP0)
				if cellP1 is None:
					cellP1 = currentCell
					r : float = -1.0
				else:
					r : float = -0.5

				actionP1 : Action = self.takeAction(cellP1, 0.0)

				# Update Q
				self.Q[currentCell.getCellNum(), actionP0.value] += learningRate * (r + gamma * self.Q[cellP1.getCellNum(), actionP1.value] - self.Q[currentCell.getCellNum(), actionP0.value])

				currentCell = cellP1
				
			# Still not founded.
		# Done.
		return founded

	"""
		@brief Get the path with already computed Q (@see learn). Can return None if no path at all has been founded.
	"""
	def getPath(self, entryCell : Cell, exitCell : Cell, maxTries : int = 10000) -> list[Action]:
		currentCell : Cell = entryCell

		pathFounded : list[Action] = []

		for i in range(0, maxTries):
			if currentCell == exitCell:
				return pathFounded

			action : Action = self.takeAction(currentCell, 0.0)
			currentCell = currentCell.getNeighbour(action)

			if currentCell is None:
				return None

			pathFounded.append(action)
		
		return None

	@staticmethod
	def pathToStr(path : list[Action]) -> str:
		outputStr = '['
		for i in range(0, len(path)):
			a : Action = path[i]
			if i > 0:
				outputStr += ', '
			outputStr += a.name
		outputStr += ']'
		return outputStr

		
"""
	@brief Class used to construct a maze.
"""
class MazeConstructor:

	shape : numpy.shape

	def __init__(self, shape : numpy.shape) -> None:
		self.shape = shape

		"""
		@brief	Update Q using Q-Learning to find a path from the entryCell to exitCell.
		@param nbLoops		Number of iterations to be done.
		@param maxTries		Number of Tries for each loop before canceling.
		@param epsilon		Value between [0.0;1.0] represent the randomness of the search.
		@param learningRate	Speed used to modify Q.
		@param gamma		Offset used to add more value to current values intead of old ones.
	"""
	def findMaze(self, nbLoops : int = 1000, maxTries : int = 20, epsilon : float = 0.25, learningRate : float = 0.01, gamma : float = 0.9) -> bool:
		Log.displayLog(f'Searching for a Maze with {nbLoops} loops and a max tries of {maxTries}.')

		maze : Maze = Maze(self.shape)
		pathFinder : PathFinder = PathFinder(maze)

		maze.reset()

		if maze.setRandomlySpecialsCells() == False:
			Log.displayError('Unable to set the specials cells.')
			return False

		while True:
			if maze.removeWallRandomly() == False:
				Log.displayError('Unable to remove any wall.')
				return False
			
			print(maze)

			# Now we have removed a wall, lets call the Learning-Q to find a solution.

			# Search path to the treasure.
			pathFinder.reset()
			if pathFinder.learn(maze.getEntryCell(), maze.getTreasureCell(), nbLoops, maxTries, epsilon, learningRate, gamma) == True:
				foundedTreasurePath : list[Action] = pathFinder.getPath(maze.getEntryCell(), maze.getTreasureCell(), maxTries)
			else:
				foundedTreasurePath : list[Action] = None

			if foundedTreasurePath is None:
				continue

			# Search path to the exit.
			pathFinder.reset()
			if pathFinder.learn(maze.getTreasureCell(), maze.getExitCell(), nbLoops, maxTries, epsilon, learningRate, gamma) == True:
				foundedExitPath : list[Action] = pathFinder.getPath(maze.getTreasureCell(), maze.getExitCell(), maxTries)
			else:
				foundedExitPath : list[Action] = None

			if foundedExitPath is None:
				continue
				
			Log.displayLog(f'Successfully founded a maze with a total path length of {len(foundedTreasurePath) + len(foundedExitPath)}.')
			Log.displayLog(f'Path entry -> treasure : {PathFinder.pathToStr(foundedTreasurePath)}.')
			Log.displayLog(f'Path treasure -> exit : {PathFinder.pathToStr(foundedExitPath)}.')
			
			return True

			


# DEBUG

mazeConstructor : MazeConstructor = MazeConstructor((4, 4))
mazeConstructor.findMaze()