import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    for idx in range(max_steps):
        ## YOUR CODE HERE
        # If current is a solution for csp then return current
        # var <- a randomly chosen conflicted variable from csp.Varaibles
        # value <- the value v for var that minimizes CONFLICTS(var, v, current, csp)
        # set var = value in current

        # Judge Conflicts
        #print(f"Sol: {solution}")
        issues = conflict_checker(solution, N)
        #print(issues)
        if not issues:
            return solution, num_steps
        
        # Measure conflicts
        issue_index = np.random.randint(len(issues))
        #print(f"Issue: {issues[issue_index]}")
        conflict_vector = conflict_measure(solution, issues[issue_index], N)
        #print(conflict_vector)

        # Select min conflicts
        best = np.where(conflict_vector == conflict_vector.min())
        #print(best[0])
        best_index = np.random.randint(len(best[0]))
        #print(f"New: {best[0][best_index]}")
        #print(f"Curr: {solution[issue_index]}")
        #plot_n_queens_solution(solution)
        solution[issues[issue_index]] = best[0][best_index]
        #print(solution[issues[issue_index]])

        num_steps += 1
        #print(f"num_steps: {num_steps}")

    return solution, num_steps


def conflict_checker(curr_assign, N):
    issue_indices = []
    diag_conflicts_up = np.zeros(shape=(2*N-1,N)) # will mark of the different diagonals
    diag_conflicts_down = np.zeros(shape=(2*N-1,N))
    row_conflicts = np.zeros(shape=(N,N))

    for i in range(N):
        # Check rows
        if row_conflicts[curr_assign[i]].any() == 1:
            issue_indices.append(i)
            for j in range(N):
                if row_conflicts[curr_assign[i], j] == 1 and j not in issue_indices:
                    issue_indices.append(j)
        else:
            row_conflicts[curr_assign[i], i] = 1

        # Check diagonals
        diag_up_diff = i - curr_assign[i] + N - 1
        diag_down_diff = i + curr_assign[i]

        #print(diag_up_diff)
        #print(diag_down_diff)

        if diag_conflicts_up[diag_up_diff].any() == 1 or diag_conflicts_down[diag_down_diff].any() == 1:
            issue_indices.append(i)
            for j in range(N):
                if diag_conflicts_up[diag_up_diff, j] == 1 and j not in issue_indices:
                    issue_indices.append(j)
                if diag_conflicts_down[diag_down_diff, j] == 1 and j not in issue_indices:
                    issue_indices.append(j)

        diag_conflicts_up[diag_up_diff, i] = 1
        diag_conflicts_down[diag_down_diff , i] = 1

    return issue_indices

def conflict_measure(curr_assign, conflict_index, N):

    #print(f"Conflict Index: {conflict_index}")

    conflict_vec = np.zeros(N, dtype=int)
    #conflict_vec[conflict_index] = 5

    for i in range(N):
        # Skip the conflicted column
        if i == conflict_index:
            continue

        #print("Row Conflict")
        conflict_vec[curr_assign[i]] += 1

        diff = abs(conflict_index-i)
        #print(f"Diff: {diff}")
        if curr_assign[i] - diff > -1: # MARKEIDJEAIJDAWI 
            #print("Diag Conflict")
            conflict_vec[curr_assign[i] - diff] += 1
        if curr_assign[i] + diff < N:
            #print("Diag Conflict")
            conflict_vec[curr_assign[i] + diff] += 1
        
    return conflict_vec

if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 1061
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    #plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    #plot_n_queens_solution(assignment_solved)
