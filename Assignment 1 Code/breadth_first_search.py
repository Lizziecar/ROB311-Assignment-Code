from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by the search
             max_frontier_size: maximum frontier size during search
    """

    # Initialize the queue with the initial state
    queue = deque([[problem.init_state]]) # sets up a queue for the BFS
    
    # Track the explored set
    explored = set() 

    # Initializes these variables
    max_frontier_size = 1
    num_nodes_expanded = 0

    while queue: # goes until empty
        # Pop the first path from the queue
        path = queue.popleft() # deque essentially
        # Get the last state from the path
        state = path[-1]

        # If this state is a goal, then we have a solution
        if state in problem.goal_states:
            return path, num_nodes_expanded, max_frontier_size

        # Mark the state as explored
        explored.add(state) # add it to the set of explored states

        # Add all the neighbours to the queue
        for action in problem.get_actions(state):
            next_state = action[1] # select the 2nd one in the list since that's the other node
            if next_state not in explored: # prevent extra work
                # Create a new path and add it to the queue
                new_path = list(path)
                new_path.append(next_state)
                queue.append(new_path)
                
                # Update extra variables
                max_frontier_size = max(max_frontier_size, len(queue))
                num_nodes_expanded += 1
    

    # If it gets here then there were no paths to goal states
    return path, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)