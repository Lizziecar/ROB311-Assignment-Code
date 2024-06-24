import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem


def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####

    # Initialize the queue with the initial state
    frontier = queue.PriorityQueue()
    parent_path = {} # track parents of nodes of best path
    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []

    # Track explored set and cost
    cost = {problem.init_state: 0}

    frontier.put((0, problem.init_state))
    
    while frontier:
        num_nodes_expanded += 1
        blah, node = frontier.get()

        if node in problem.goal_states: # solution found
            node = problem.goal_states[0]
            while node is not problem.init_state: # create path from goal to initial
                path.append(node)
                node = parent_path[node]
            
            path.append(problem.init_state) # append initial
            path = path[::-1] # reverse path

            return path, num_nodes_expanded, max_frontier_size
        
        for action in problem.get_actions(node):
            child = action[1]
            if child not in cost or (cost[node] + problem.heuristic(child) < cost[child]): # need to update collection of costs
                cost[child] = 1 + cost[node] # set new cost
                frontier.put((cost[node] + 1 + problem.heuristic(child), child))
                parent_path[child] = node # record 

        max_frontier_size = max(max_frontier_size, frontier.qsize())

    return path, num_nodes_expanded, max_frontier_size


def run_test():
    
    # Set up axis
    p_occ_axis = np.arange(0.1,0.95,0.05)

    # Set up outputs
    solved1, solved2, solved3 = np.zeros(len(p_occ_axis))
    average_nodes1, average_nodes2, average_nodes3 = np.zeros(len(p_occ_axis))

    ind = 0
    runs = 100
    for i in p_occ_axis:
        solve1, solve2, solve3, avg1, avg2, avg3 = 0
        for j in range(n_runs):
            # Create random grid problems
            problem1 = get_random_grid_problem(i,20,20)
            problem2 = get_random_grid_problem(i,100,100)
            problem3 = get_random_grid_problem(i,500,500)

            # Solve Problems using A star
            path1, expand1, frontier1 = a_star_search(problem1)
            path2, expand2, frontier2 = a_star_search(problem2)
            path3, expand3, frontier3 = a_star_search(problem3)

            # Check solutions
            if(problem1.check_solution(path1)): solve1 += 1 
            if(problem2.check_solution(path2)): solve2 += 1 
            if(problem3.check_solution(path3)): solve3 += 1 
            
            # Measure number of nodes expaneded
            avg1 += expand1
            avg2 += expand2
            avg3 += expand3
        
        # Take percentages
        solve1 /= 100
        solve2 /= 100
        solve3 /= 100
        avg1 /=100
        avg2 /= 100
        avg3 /=100

        solved1[ind] = solve1
        solved2[ind] = solve2
        solved3[ind] = solve3

        average_nodes1[ind] = avg1
        average_nodes2[ind] = avg2
        average_nodes3[ind] = avg3
        ind += 1
    
    # graph everything
    fig, ax = plt.subplots(3,2)
    ax[0,0].plot(p_occ_axis, solved1)
    ax[0,0].set_title("N = 20, solved")
    ax[0,1].plot(p_occ_axis, average_nodes1)
    ax[0,1].set_title("N = 20, average")
    ax[1,0].plot(p_occ_axis, solved2)
    ax[1,0].set_title("N = 100, solved")
    ax[1,1].plot(p_occ_axis, average_nodes2)
    ax[1,1].set_title("N = 100, average")
    ax[2,0].plot(p_occ_axis, solved3)
    ax[2,0].set_title("N = 500, solved")
    ax[2,1].plot(p_occ_axis, average_nodes3)
    ax[2,1].set_title("N = 500, average")
    plt.show()

def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.3 # pure guess
    transition_end_probability = 0.4 # pure guess
    peak_nodes_expanded_probability = 0.4 # pure guess
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS
    run_test()