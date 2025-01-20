import numpy as np

class RegressionTree:
    class TreeNode:
        def __init__(
                self, 
                parerent_node = None,
                left_child = None, 
                right_child = None, 
                depth: int = 0,
                sample_inputs: np.ndarray = None, 
                sample_outputs: np.ndarray = None) -> None:
            self.parent_node = parerent_node
            self.left_child = left_child
            self.right_child = right_child
            self.depth = depth
            self.sample_inputs = sample_inputs
            self.sample_outputs = sample_outputs
            self.split_feature = None
            self.split_threshold = None
            self.prediction = None

        def search_optimal_split(self, loss_func, min_sample_per_leaf) -> bool:
            """
            Search the best split for this node and set attributes accordingly

            Parameters:
            loss_func: function
                a fuction that takes 2 arguments, an array of expected values, and an array or value for predictions, and returns a value to be minimised

            Output: bool
                true if the split succeded, false otherwise
                if false, the node is a leaf
            """
            _, number_of_features = self.sample_inputs.shape
            min_loss = None
            samples = len(self.sample_inputs)

            for feature_index in range(number_of_features):
                feature_column = self.sample_inputs[:, feature_index]
                sorted_index = feature_column.argsort()

                feature_column = feature_column[sorted_index]
                output_column = self.sample_outputs[sorted_index]

                _, unique_thresholds_indices = np.unique(feature_column, return_index = True) # O(nlogn) could be done in O(N) with numpy tricks, hopefully

                for i in range(1, len(unique_thresholds_indices)):
                    threshold_index = unique_thresholds_indices[i]

                    if threshold_index < min_sample_per_leaf or samples - threshold_index < min_sample_per_leaf:
                        continue

                    # if the feature and thresholds are sorted we could reuse previous work as the threshold is incremented
                    lower_side_outputs = output_column[:threshold_index]
                    higher_side_outputs = output_column[threshold_index:]

                    pred = np.empty(samples)
                    pred[:threshold_index] = lower_side_outputs.mean()
                    pred[threshold_index:] = higher_side_outputs.mean()

                    current_split_loss = loss_func(output_column, pred)

                    if min_loss is None or current_split_loss < min_loss:
                        min_loss = current_split_loss
                        self.split_feature = feature_index
                        self.split_threshold = feature_column[unique_thresholds_indices[i - 1]]

            if min_loss is None:    # no split was found
                self.prediction = self.sample_outputs.mean()
                # print(f"no min was found, depth {self.depth}")
                return False

            left_side = self.sample_inputs[:, self.split_feature] <= self.split_threshold
            right_side = self.sample_inputs[:, self.split_feature] > self.split_threshold

            self.left_child = RegressionTree.TreeNode(
                depth = self.depth + 1,
                parerent_node = self,
                sample_inputs = self.sample_inputs[left_side],
                sample_outputs = self.sample_outputs[left_side]
            )

            self.right_child = RegressionTree.TreeNode(
                depth = self.depth + 1,
                parerent_node = self,
                sample_inputs = self.sample_inputs[right_side],
                sample_outputs = self.sample_outputs[right_side]
            )

            # print(f"depth {self.depth} split feat {self.split_feature} split threshold {self.split_threshold}, left {len(self.left_child.sample_inputs)}, right {len(self.right_child.sample_inputs)}")

            return True

    def __init__(
            self, 
            loss_func,
            max_depth:              int | None  = None, 
            min_samples_split:      int = 2,
            min_samples_leaf:       int = 1,
            postpruning:            bool = False,
            validation_fraction:    float = 0.1
        ) -> None:
        """
        Create a regression tree

        Parameters:
        loss_func: function
            a fuction that takes 2 arguments, an array of expected values, and an array or value for predictions, and returns a value to be minimised
        max_depth: int | None (default: None)
            the maximum depth of the tree, unlimited depth if None
        min_samples_split: int (default: 2)
            the minimum number of samples in an internal node for it to be split
        min_samples_leaf: int (default: 1)
            the minimum number of samples in a leaf
        postpruning: bool (default: False)
            whether to use Reduced Error Pruning (REP)
        validation_fraction: float (default: 0.1)
            the size of the validation fraction to use for REP
        """
        self.__loss = loss_func
        self.__max_depth = max_depth if max_depth is not None and max_depth >= 1 else None
        self.__min_sample_split = min_samples_split if min_samples_split >= 2 else 2
        self.__min_sample_leaf = min_samples_leaf if min_samples_leaf >= 1 else 1
        self.__postpruing = postpruning
        self.__validation_fraction = validation_fraction
        self.__tree_root = RegressionTree.TreeNode()
        self.__validation_inputs = None
        self.__validation_outputs = None
        self.__validation_size = None

    def __postprune(self) ->None:
        """
        Apply bottom-up Reduced Error Pruning
        """

        #postorder traversal
        current_node = self.__tree_root
        visited = set()

        while current_node is not None and current_node not in visited:  
            if current_node.left_child is not None and current_node.left_child not in visited:
                current_node = current_node.left_child
                continue

            if current_node.right_child is not None and current_node.right_child not in visited:
                current_node = current_node.right_child
                continue

            if current_node.prediction is None:     # not a leaf
                if current_node.left_child.prediction is not None and current_node.right_child.prediction is not None:
                    #both children are leaves, try to transform current node into leaf
                    parent_mean = current_node.sample_outputs.mean()
                    parent_pred = np.full(self.__validation_size, parent_mean)

                    left_mask = self.__validation_inputs[:, current_node.split_feature] <= current_node.split_threshold
                    right_mask = np.bitwise_not(left_mask)

                    children_pred = left_mask * current_node.left_child.prediction + right_mask * current_node.right_child.prediction

                    parent_loss = self.__loss(self.__validation_outputs, parent_pred)
                    children_loss = self.__loss(self.__validation_outputs, children_pred)

                    if parent_loss < children_loss:
                        # delete leaves
                        current_node.left_child = None
                        current_node.right_child = None
                        current_node.prediction = parent_mean
                        
            visited.add(current_node)
            current_node = current_node.parent_node


    def fit(self, train_inputs, train_outputs) -> None:
        """
        Trains / Retrains the tree model

        Parameters:
        train_inputs: numpy.ndarray | ArrayLike
            2 dimensional array, containging the training input features for the trianing examples
        train_outputs: numpy.ndarrat | ArrayLike
            1 dimenstional array, containing the expected outputs for the input data
        """

        assert len(train_inputs) == len(train_outputs)
        self.__tree_root.depth = 1

        if self.__postpruing:
            train_samples = int(len(train_inputs) * (1.0 - self.__validation_fraction))
            self.__validation_size = len(train_inputs) - train_samples
            self.__validation_inputs = train_inputs[train_samples:]
            self.__validation_outputs = train_outputs[train_samples:]
            self.__tree_root.sample_inputs = train_inputs[:train_samples]
            self.__tree_root.sample_outputs = train_outputs[:train_samples]

        else:
            self.__tree_root.sample_inputs = train_inputs
            self.__tree_root.sample_outputs = train_outputs

        node_stack = [self.__tree_root]

        while len(node_stack) > 0:  #run until all the leaves were reached
            current_node = node_stack.pop()

            if current_node.depth == self.__max_depth:
                current_node.prediction = current_node.sample_outputs.mean()
                continue

            if len(current_node.sample_inputs) < self.__min_sample_split:
                current_node.prediction = current_node.sample_outputs.mean()
                continue

            success = current_node.search_optimal_split(self.__loss, self.__min_sample_leaf)

            if success:
                node_stack.append(current_node.left_child)
                node_stack.append(current_node.right_child)

        if self.__postpruing:
            self.__postprune()

    def predict(self, inputs) -> np.ndarray:
        """
        Predict the values of the inputs

        Parameters:
        inputs: ArrayLike
            2D array containing the input features
        
        Output: numpy.ndarray
            1D array containing the predicted values
        """

        predictions = np.zeros(len(inputs))

        for sample_index in range(len(inputs)):
            sample = inputs[sample_index]
            current_node = self.__tree_root

            while current_node.prediction is None:
                if sample[current_node.split_feature] <= current_node.split_threshold:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child

            predictions[sample_index] = current_node.prediction

        return predictions
            