from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by the search
                 max_frontier_size: maximum frontier size during search
        """
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    # Set queues
    queue_start = deque([[problem.init_state]])
    queue_goal = deque([[problem.goal_states[0]]])

    # Set empty sets
    explored_start = set([problem.init_state])
    explored_goal = set([problem.goal_states[0]])

    # Initializes these variables
    max_frontier_size = 1
    num_nodes_expanded = 0

    while queue_start and queue_goal:
        path_start = queue_start.popleft() # array
        path_goal = queue_goal.popleft() # array

        state_start = path_start[-1]
        state_goal = path_goal[-1]

        explored_start.add(state_start)
        explored_goal.add(state_goal)

        print(state_start)
        print(state_goal)

        # Add all the neighbours to the queue
        for action in problem.get_actions(state_start):
            next_state = action[1] # select the 2nd one in the list since that's the other node
            if next_state not in explored_start: # prevent extra work
                # Create a new path and add it to the queue
                new_path = list(path_start)
                new_path.append(next_state)
                queue_start.append(new_path)
               
                # Check for intersection
                for paths_start, paths_goal in zip(queue_start, queue_goal):
                    #print(f"Looking for {next_state} in {paths_goal}")
                    if next_state in paths_goal: # if intersection
                        new_path = []
                        for state in paths_goal: # go through paths_goal
                            #print(state)
                            new_path.append(state)
                            #print(new_path)
                            if state == next_state: # go until you find the state
                                break
                        
                        path_final = path_start + new_path[::-1]
                        return path_final, num_nodes_expanded, max_frontier_size
                
                # Update extra variables
                max_frontier_size = max(max_frontier_size, len(queue_start))
                num_nodes_expanded += 1

        for action in problem.get_actions(state_goal):
            next_state = action[1] # select the 2nd one in the list since that's the other node
            if next_state not in explored_goal: # prevent extra work
                # Create a new path and add it to the queue
                new_path = list(path_goal)
                new_path.append(next_state)
                queue_goal.append(new_path)

                # Check for intersection
                for paths_start, paths_goal in zip(queue_start, queue_goal):
                    #print(f"Looking for {next_state} in {paths_start}")
                    if next_state in paths_start: # if intersection but other way
                        #print(f"Path Goal: {path_goal}")
                        #print(f"Path start: {paths_start}")
                        new_path = []
                        for state in paths_start: # go through paths_goal
                            #print(state)
                            new_path.append(state)
                            if state == next_state: # go until you find the state
                                break
                        path_final = new_path + path_goal[::-1]
                        return path_final, num_nodes_expanded, max_frontier_size

                # Update extra variables
                max_frontier_size = max(max_frontier_size, len(queue_goal))
                num_nodes_expanded += 1


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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print("\n")
    
    
    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print("\n")

    goal_states = [15]
    init_state = 0
    V = np.arange(0, 16)
    E = np.array([[0, 1],
              [1, 2],
              [2, 3],
              [3, 4],
              [4, 5],
              [5, 6],
              [6, 7],
              [7, 8],
              [8, 9],
              [9, 10],
              [10, 11],
              [11, 12],
              [12, 13],
              [13, 14],
              [14, 15],
              [0, 5],
              [5, 10],
              [10, 15],
              [10,15],
              [2,15],
              [3,15]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print("Length of path: {path}")
    print("\n")
    