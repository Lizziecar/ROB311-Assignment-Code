import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    greedy_init = np.zeros(N, dtype=int)
    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)

    ### YOUR CODE GOES HERE
    # Constraints but math:
    # No entry can be the same number
    # No entry can have sucssive or reverse succesive value i.e. [1,2] will create a conflict
    # Only 1 per column

    for i in range(N):
        if i == 0:
            continue
    
        conflict_vec = conflict_check(greedy_init, i, N)

        # Select min conflicts
        best = np.where(conflict_vec == conflict_vec.min())
        best_index = np.random.randint(len(best[0]))
        greedy_init[i] = best[0][best_index]

    print(conflict_checker(greedy_init, N))

    return greedy_init

def conflict_check(curr_assign, j, N):
    curr_assign_true = curr_assign[0:j]
    conflict_vec = np.zeros(N, dtype=int)

    offset = j

    for assignment in curr_assign_true:
        #print(assignment)
        #print(offset)
        conflict_vec[assignment] += 1
        if assignment > offset:
            conflict_vec[assignment-offset] +=1
        if assignment < N-offset:
            conflict_vec[assignment+offset] +=1
        offset -= 1

    #print(conflict_vec)  
    return conflict_vec

def conflict_checker(curr_assign, N):
    issue_indices = []
    diag_conflicts_up = np.zeros(shape=(2*N-1,N)) # will mark of the different diagonals
    diag_conflicts_down = np.zeros(shape=(2*N-1,N))
    row_conflicts = np.zeros(shape=(N,N))

    '''
    for i in range(N):
        #print(f"I: {i}\n")
        for j in range(1,N-i):
            #print(f"I: {i} value: {curr_assign[i]}\nJ: {i+j} value: {curr_assign[i+j]}\n")
            if curr_assign[i] == curr_assign[i+j] or abs(curr_assign[i+j] - curr_assign[i]) == j:
                #print("Conflict")
                if i not in issue_indices:
                    issue_indices.append(i)
                if i+j not in issue_indices:
                    issue_indices.append(i+j)
    '''


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

from support import plot_n_queens_solution

if __name__ == '__main__':
    # You can test your code here
    game = initialize_greedy_n_queens(6)
    print(f"\n{game}")
    plot_n_queens_solution(game)
    pass
