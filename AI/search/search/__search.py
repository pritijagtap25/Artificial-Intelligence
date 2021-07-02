# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    
    current_position = problem.getStartState()
    stack = Stack()
    stack.push(current_position)
    previous_position = None
    explored_position = [(current_position)]
    
    visited_nodes = list()
    visited_nodes.append({'current_position': current_position, 'previous_position': None, 'action': None, 'traveled': False})
    
    while not stack.isEmpty():
        current_position = stack.pop()
        explored_position.append((current_position))
        current_location = list()
        for node in visited_nodes:
                if node['current_position'] == current_position:
                        current_location.append(node)
        if len(current_location) > 1:
            for node in current_location:
                if node['previous_position'] == previous_position:
                    node['traveled'] = True
        else:
            current_location[0]['traveled'] = True

        previous_position = current_position

        if problem.isGoalState(current_position):
            break

        successor_state = problem.getSuccessors(current_position)
        for state in successor_state:
            if state[0] not in explored_position:
                visited_nodes.append({'current_position': state[0], 'previous_position': current_position, 'action': state[1], 'traveled': False})
                stack.push(state[0])
    path = []
    current_path = visited_nodes[-1]
    while current_path:
        print "current_path = ", current_path
        if not current_path['previous_position']:
            break
        path.insert(0, current_path['action'])
        current_path = next((element for element in visited_nodes if element['current_position'] == current_path['previous_position'] and element['traveled'] ), None)

    return path
		
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    queue = Queue()
    current_position = problem.getStartState()
    queue.push(current_position)
    explored_position = list()
    visited_nodes = list()
    visited_nodes.append({'current_position': current_position, 'previous_position': None, 'action': None, 'traveled': False})
    
    current_path = None
    while not queue.isEmpty():
            current_position = queue.pop()
            if current_position in explored_position:
                    continue
            
            explored_position.append((current_position))
            current_node = next((element for element in visited_nodes if element['current_position'] == current_position), None)
            current_node['traveled'] = True
            
            if problem.isGoalState(current_position):
                    current_path = current_node
                    break
            
            next_state = problem.getSuccessors(current_position)
            for state in next_state:
                if state[0] not in explored_position:
                    visited_nodes.append({'current_position':state[0],'previous_position':current_position,'action':state[1],'traveled':False})
                    queue.push(state[0])
				
    path = []
    while current_path:
        if not current_path['previous_position']:
            break
        path.insert(0, current_path['action'])
        current_path = next((element for element in visited_nodes if element['current_position'] == current_path['previous_position'] and element['traveled'] ), None)

    return path
				
	#util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    current_position = problem.getStartState()
    explored_position = list()
    
    visited_nodes = list()
    visited_nodes.append({'current_position': current_position, 'previous_position': None, 'action': None, 'traveled': False, 'cost': 0})
    
    stack = PriorityQueue()
    stack.push(current_position,0)
    
    current_path = None
    while not stack.isEmpty():
        current_position = stack.pop()
        if current_position in explored_position:
                continue
        explored_position.append((current_position))
        possible_nodes = list()
        for node in visited_nodes:
            if node['current_position'] == current_position:
                possible_nodes.append(node)
            
        if len(possible_nodes)>1:
            node_s = possible_nodes[0]
            for node in possible_nodes:
                if node_s['cost'] > node['cost']:
                    node_s = node
            node_current = node_s
        else:
            node_current = possible_nodes[0]
        
        node_current['traveled'] = True
        if problem.isGoalState(current_position):
	    current_path = node_current
            break
			
        next_state = problem.getSuccessors(current_position)
        for state in next_state:
            if state[0] not in explored_position:
                cost = state[2] + node_current['cost']
                visited_nodes.append({'current_position': state[0], 'previous_position': current_position, 'action': state[1], 'traveled': False, 'cost': cost })
                stack.push(state[0],cost)
                        
    current_location = list()
    while current_path:
        if not current_path['previous_position']:
                break
        current_location.insert(0,current_path['action'])
        possible_path = list()
        
        for node in visited_nodes:
            if node['current_position'] == current_path['previous_position'] and node['traveled']:
                possible_path.append(node)
                                        
        if len(possible_path) > 1:
            current_path = possible_path[0]
            for path in potential_path:
                if current_path['cost'] > path['cost']:
                    current_path = path
                                                
        else:
            current_path = possible_path[0]
					
	return current_location
            

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    from util import PriorityQueue
    current_position = problem.getStartState()
    explored_position = list()
    visited_nodes = list()
    visited_nodes.append({'current_position': current_position, 'previous_position': None, 'action': None, 'traveled': False, 'cost': 0})
    
    stack = PriorityQueue()
    stack.push(current_position,0)
    
    current_path = None
    while not stack.isEmpty():
        current_position = stack.pop()
        if current_position in explored_position:
                continue
        explored_position.append((current_position))
        possible_nodes = list()
        for node in visited_nodes:
            if node['current_position'] == current_position:
                possible_nodes.append(node)
            
        if len(possible_nodes)>1:
            node_s = possible_nodes[0]
            for node in possible_nodes:
                if node_s['cost'] > node['cost']:
                    node_s = node
            node_current = node_s
        else:
            node_current = possible_nodes[0]
        
        node_current['traveled'] = True
        if problem.isGoalState(current_position):
	    current_path = node_current
            break
			
        next_state = problem.getSuccessors(current_position)
        for state in next_state:
            if state[0] not in explored_position:
                cost = state[2] + node_current['cost']
                heuristic_cost = cost + heuristic(state[0],problem)
                visited_nodes.append({'current_position': state[0], 'previous_position': current_position, 'action': state[1], 'traveled': False, 'cost': cost })
                stack.push(state[0],cost)
                        
    current_location = list()
    while current_path:
        if not current_path['previous_position']:
                break
        current_location.insert(0,current_path['action'])
        possible_path = list()
        
        for node in visited_nodes:
            if node['current_position'] == current_path['previous_position'] and node['traveled']:
                possible_path.append(node)
                                        
        if len(possible_path) > 1:
            current_path = possible_path[0]
            for path in potential_path:
                if current_path['cost'] > path['cost']:
                    current_path = path
                                                
        else:
            current_path = possible_path[0]
					
	return current_location
                        
                            



        
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
