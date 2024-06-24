import numpy as np
# DO NOT ADD TO OR MODIFY ANY IMPORT STATEMENTS


def dt_entropy(goal, examples):
    """
    Compute entropy over discrete random varialbe for decision trees.
    Utility function to compute the entropy (which is always over the 'decision'
    variable, which is the last column in the examples).

    :param goal: Decision variable (e.g., WillWait), cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the entropy of the decision variable, given examples.
    """
    # INSERT YOUR CODE HERE.
    entropy = 0.
    num_examples = len(examples)

    if num_examples == 0:
        return 0;
        
    n = len(goal[1])
    counts = np.zeros(n)

    # Count instances
    for example in examples:
        #print(example[-1])
        counts[example[-1]] += 1

    # Calculate probabilities
    probs = counts/num_examples

    # Calculate entropy
    for probs in probs:
        if probs == 0: # make sure no NaN issues
            continue
        entropy -= probs*np.log2(probs)

    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0
    return entropy


def dt_cond_entropy(attribute, col_idx, goal, examples):
    """
    Compute the conditional entropy for attribute. Utility function to compute the conditional entropy (which is always
    over the 'decision' variable or goal), given a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the conditional entropy, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    cond_entropy = 0.0

    num_examples = len(examples)

    if num_examples == 0:
        return 0;

    n = len(goal[1])
    m = len(attribute[1])
    counts = np.zeros(shape=(m,n))

    num_examples_with_attribute = np.zeros(m)
    num_examples_with_goal = np.zeros(n)

    # Number of classes in training example
    for example in examples:

        #num_examples_with_attribute[example[col_idx]] += 1
        #num_examples_with_goal[example[len(example)-1]] += 1
        counts[example[col_idx], example[len(example)-1]] += 1

    #print(counts)

    # Calculate number of goal and attribute examples
    num_examples_with_attribute = np.sum(counts, axis=1)
    num_examples_with_goal = np.sum(counts,axis=0)

    # Calcualte attribute probability
    att_prob = num_examples_with_attribute/num_examples

    # Calculate conditional prob
    probs = np.zeros(shape=(m,n))

    for i in range(m):
        for j in range(n):
            if num_examples_with_attribute[i] > 0:
                probs[i,j] = counts[i,j] / num_examples_with_attribute[i]
            else: 
                probs[i,j] = 0

    #print(num_examples_with_attribute)
    #print(probs) 

    # Calculate all logs:
    log_pik = np.zeros(shape=(m,n))
    
    for k in range(m):
        for j in range(n):
            if probs[k, j] == 0:
                log_pik[k,j] = 0
            else:
                log_pik[k,j] = np.log2(probs[k,j])

    # Calculate Remainder
    cond_entropy = -np.sum(att_prob*np.sum(probs*log_pik, axis=-1))

    return cond_entropy


def dt_info_gain(attribute, col_idx, goal, examples):
    """
    Compute information gain for attribute.
    Utility function to compute the information gain after splitting on attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the information gain, given the attribute and examples.

    """
    # INSERT YOUR CODE HERE.
    info_gain = 0.

    entropy = dt_entropy(goal, examples)
    cond_entropy = dt_cond_entropy(attribute, col_idx, goal, examples)

    info_gain = entropy - cond_entropy

    return info_gain


def dt_intrinsic_info(attribute, col_idx, examples):
    """
    Compute the intrinsic information for attribute.
    Utility function to compute the intrinsic information of a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the intrinsic information for the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0

    num_examples = len(examples)
    intrinsic_info = 0.

    if num_examples == 0:
        return 0;
        
    n = len(attribute[1])
    counts = np.zeros(n)

    # Count instances
    for example in examples:
        counts[example[col_idx]] += 1

    # Calculate probabilities
    probs = counts/num_examples

    # Calculate entropy
    for probs in probs:
        if probs == 0: # make sure no NaN issues
            continue
        intrinsic_info -= probs*np.log2(probs)

    return intrinsic_info


def dt_gain_ratio(attribute, col_idx, goal, examples):
    """
    Compute information gain ratio for attribute.
    Utility function to compute the gain ratio after splitting on attribute. Note that this is just the information
    gain divided by the intrinsic information.
    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the gain ratio, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Avoid NaN examples by treating 0.0/0.0 = 0.0
    gain_ratio = 0.

    information_gain = dt_info_gain(attribute, col_idx, goal, examples)
    intrinsic_info = dt_intrinsic_info(attribute, col_idx, examples)

    if intrinsic_info > 0:
        gain_ratio = information_gain / intrinsic_info
    else:
        gain_ratio = 0

    return gain_ratio


def learn_decision_tree(parent, attributes, goal, examples, score_fun):
    """
    Recursively learn a decision tree from training data.
    Learn a decision tree from training data, using the specified scoring function to determine which attribute to split
    on at each step. This is an implementation of the algorithm on pg. 702 of AIMA.

    :param parent: Parent node in tree (or None if first call of this algorithm).
    :param attributes: Attributes avaialble for splitting at this node.
    :param goal: Goal, decision variable (classes/labels).
    :param examples: Subset of examples that reach this point in the tree.
    :param score_fun: Scoring function used (dt_info_gain or dt_gain_ratio)
    :return: Root node of tree structure.
    """
    # YOUR CODE GOES HERE
    node = None
    # 1. Do any examples reach this point?

    # No examples, return pluirty
    #print(examples)
    if len(examples) == 0:
        label = plurality_value(goal, parent.examples)
        #print(f"No Examples: {label}")
        return TreeNode(parent, None, None, True, label)

    # 2. Or do all examples have the same class/label? If so, we're done!
    
    # All the same classification, return classification
    for i in range(len(goal[1])):
        list_goal = []
        for example in examples:
            goal_val = example[len(example)-1]
            #print(goal_val)
            if goal_val == i:
                list_goal.append(goal_val)
            #print(list_goal)
        
        if len(list_goal) == len(examples):
            #print(f"All classes: {list_goal[0]}")
            return TreeNode(parent, None, None, True, list_goal[0])
        
    # 3. No attributes left? Choose the majority class/label.

    # No attributes left, return plurity-value
    if len(attributes) == 0:
        label = plurality_value(goal, examples)
        #print(f"No Attributes: {label}")
        return TreeNode(parent, None, None, True, label)

    # 4. Otherwise, need to choose an attribute to split on, but which one? Use score_fun and loop over attributes!

    # Best score:
    attribute_scores = np.zeros(len(attributes))
    best_score = 0
    best_index = 0
    for i in range(len(attributes)):
        #print(attributes[i])
        #print(f"Example: {examples[0]}")
        score = score_fun(attributes[i], i, goal, examples)
        print(f"{i}: {score}")
        attribute_scores[i] = score
        if i == 0:
            best_score = score
        elif score > best_score:
            best_score = score
            best_index = i

    # NOTE: to pass the Autolab tests, when breaking ties you should always select the attribute with the smallest (i.e.
    # leftmost) column index!

    # Create a new internal node using the best attribute, something like:
    #(f"Best Index: {best_index}")
    #print(f"Max Index: {len(attribute_scores)}")
    node = TreeNode(parent, attributes[best_index], examples, False, goal)

    # Now, recurse down each branch (operating on a subset of examples below).

    # For each attribute value
    for i in range(len(attributes[best_index][1])):
        #print(f"Attribute Value: {i}")
        exs = [] 
        
        # remove used attribute from attributes
        #print(f"Best Attribute: {attributes[best_index]}")
        new_attributes = attributes[:best_index] + attributes[best_index+1:]
        
        #print(f"Prev: {attributes}")
        #print(f"New: {new_attributes}")

        # get all the examples that correspond to that value
        for example in examples:
            if example[best_index] == i:
                exs.append(np.copy(example))

        if exs:
            #print(f"Examples: {exs}")
            exs = np.delete(exs, best_index, 1)

        # Make a subtree
        subtree = learn_decision_tree(node, new_attributes, goal, exs, score_fun)

        # Add a branch to the tree:
        node.branches.append(subtree)

    # You should append to node.branches in this recursion

    return node

def plurality_value(goal: tuple, examples: np.ndarray) -> int:
    """
    Utility function to pick class/label from mode of examples (see AIMA pg. 702).
    :param goal: Tuple representing the goal
    :param examples: (n, m) array of examples, each row is an example.
    :return: index of label representing the mode of example labels.
    """
    vals = np.zeros(len(goal[1]))

    # Get counts of number of examples in each possible attribute class first.
    for i in range(len(goal[1])):
        vals[i] = sum(examples[:, -1] == i)

    return np.argmax(vals)


class TreeNode:
    """
    Class representing a node in a decision tree.
    When parent == None, this is the root of a decision tree.
    """
    def __init__(self, parent, attribute, examples, is_leaf, label):
        # Parent node in the tree
        self.parent = parent
        # Attribute that this node splits on
        self.attribute = attribute
        # Examples used in training
        self.examples = examples
        # Boolean representing whether this is a leaf in the decision tree
        self.is_leaf = is_leaf
        # Label of this node (important for leaf nodes that determine classification output)
        self.label = label
        # List of nodes
        self.branches = []

    def query(self, attributes: np.ndarray, goal, query: np.ndarray) -> (int, str):
        """
        Query the decision tree that self is the root of at test time.

        :param attributes: Attributes available for splitting at this node
        :param goal: Goal, decision variable (classes/labels).
        :param query: A test query which is a (n,) array of attribute values, same format as examples but with the final
                      class label).
        :return: label_val, label_txt: integer and string representing the label index and label name.
        """
        node = self
        while not node.is_leaf:
            b = node.get_branch(attributes, query)
            node = node.branches[b]

        print(node.label)
        return node.label, goal[1][node.label]

    def get_branch(self, attributes: list, query: np.ndarray):
        """
        Find attributes in a set of attributes and determine which branch to use (return index of that branch)

        :param attributes: list of attributes
        :param query: A test query which is a (n,) array of attribute values.
        :return:
        """
        for i in range(len(attributes)):
            if self.attribute[0] == attributes[i][0]:
                return query[i]
        # Return None if that attribute can't be found
        return None

    def count_tree_nodes(self, root=True) -> int:
        """
        Count the number of decision nodes in a decision tree.
        :param root: boolean indicating if this is the root of a decision tree (needed for recursion base case)
        :return: number of nodes in the tree
        """
        num = 0
        for branch in self.branches:
            num += branch.count_tree_nodes(root=False) + 1
        return num + root


if __name__ == '__main__':
    # Example use of a decision tree from AIMA's restaurant problem on page (pg. 698)
    # Each attribute is a tuple of 2 elements: the 1st is the attribute name (a string), the 2nd is a tuple of options
    a0 = ('Alternate', ('No', 'Yes'))
    a1 = ('Bar', ('No', 'Yes'))
    a2 = ('Fri-Sat', ('No', 'Yes'))
    a3 = ('Hungry', ('No', 'Yes'))
    a4 = ('Patrons', ('None', 'Some', 'Full'))
    a5 = ('Price', ('$', '$$', '$$$'))
    a6 = ('Raining', ('No', 'Yes'))
    a7 = ('Reservation', ('No', 'Yes'))
    a8 = ('Type', ('French', 'Italian', 'Thai', 'Burger'))
    a9 = ('WaitEstimate', ('0-10', '10-30', '30-60', '>60'))
    attributes = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    # The goal is a tuple of 2 elements: the 1st is the decision's name, the 2nd is a tuple of options
    goal = ('WillWait', ('No', 'Yes'))

    # Let's input the training data (12 examples in Figure 18.3, AIMA pg. 700)
    # Each row is an example we will use for training: 10 features/attributes and 1 outcome (the last element)
    # The first 10 columns are the attributes with 0-indexed indices representing the value of the attribute
    # For example, the leftmost column represents the attribute 'Alternate': 0 is 'No', 1 is 'Yes'
    # Another example: the 3rd last column is 'Type': 0 is 'French', 1 is 'Italian', 2 is 'Thai', 3 is 'Burger'
    # The 11th and final column is the label corresponding to the index of the goal 'WillWait': 0 is 'No', 1 is 'Yes'
    examples = np.array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
                         [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
                         [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
                         [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
                         [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                         [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
                         [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
                         [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                         [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1]])


    print(dt_cond_entropy(a6, 6, goal, examples))


    # Build your decision tree using dt_info_gain as the score function
    tree = learn_decision_tree(None, attributes, goal, examples, dt_info_gain)
    print(tree)

    
    # Query the tree with an unseen test example: it should be classified as 'Yes'
    test_query = np.array([0, 0, 1, 1, 2, 0, 0, 0, 2, 3])
    _, test_class = tree.query(attributes, goal, test_query)
    print("Result of query: {:}".format(test_class))

    # Repeat with dt_gain_ratio:
    tree_gain_ratio = learn_decision_tree(None, attributes, goal, examples, dt_gain_ratio)
    # Query this new tree: it should also be classified as 'Yes'
    _, test_class = tree_gain_ratio.query(attributes, goal, test_query)
    print("Result of query with gain ratio as score: {:}".format(test_class))
