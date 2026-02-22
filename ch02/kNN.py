"""k-Nearest Neighbors (kNN) classification algorithm implementation."""

import operator
import os

import numpy as np
import numpy.typing as npt


def create_data_set() -> tuple[npt.NDArray[np.float64], list[str]]:
    """Create a simple dataset for testing."""
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify0(
    in_x: npt.NDArray[np.float64],
    data_set: npt.NDArray[np.float64],
    labels: list[str],
    k: int,
) -> str:
    """Classify a sample using kNN algorithm.

    Args:
        in_x: Input sample to classify.
        data_set: Training dataset.
        labels: Labels for training samples.
        k: Number of neighbors to consider.

    Returns:
        Predicted label for the input sample.
    """
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat**2
    sq_distances: npt.NDArray[np.float64] = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5
    sorted_dist_indicies: npt.NDArray[np.int64] = distances.argsort()
    class_count: dict[str, int] = {}
    for i in range(k):
        vote_ilabel: str = labels[sorted_dist_indicies[i]]
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
    sorted_class_count = sorted(
        class_count.items(), key=operator.itemgetter(1), reverse=True
    )
    return sorted_class_count[0][0]


def file2matrix(filename: str):
    """Read a file and convert it to a matrix.

    Args:
        filename: Path to the input file.

    Returns:
        Tuple of feature matrix and label vector.
    """
    with open(filename, encoding="utf-8") as fr:
        array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)

    return_mat = np.zeros((number_of_lines, 3))
    class_label_vector = []

    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split("\t")
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set: npt.NDArray[np.float64]):
    """Normalize the dataset using min-max normalization.

    Args:
        data_set: Input dataset to normalize.

    Returns:
        Tuple of normalized dataset, ranges, and minimum values.
    """
    min_vals = data_set.min()
    max_vals = data_set.max()
    ranges = max_vals - min_vals

    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    """Test the classifier on dating website dataset."""
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix("./data/datingTestSet2.txt")
    norm_mat, _ranges, _min_values = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(
            norm_mat[i, :],
            norm_mat[num_test_vecs:m, :],
            dating_labels[num_test_vecs:m],
            3,
        )
        print(
            f"the classifier came back with: {classifier_result}, "
            f"the real answer is: {dating_labels[i]}"
        )
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print(f"the error count is: {error_count}")
    print(f"the total error rate is: {error_count / float(num_test_vecs)}")
    print(f"the test set size is: {num_test_vecs}")


def img2vector(filename: str) -> npt.NDArray[np.float64]:
    """Convert an image file to a feature vector.

    Args:
        filename: Path to the image file.

    Returns:
        1x1024 feature vector.
    """
    return_vect = np.zeros((1, 1024))
    with open(filename, encoding="utf-8") as fr:
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def handwriting_class_test():
    """Test the handwriting recognition classifier."""
    training_dir = "./data/trainingDigits"
    test_dir = "./data/testDigits"
    hw_labels = []
    training_file_list = os.listdir(training_dir)
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split(".")[0]
        class_num_str = int(file_str.split("_")[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector(f"{training_dir}/{file_name_str}")
    test_file_list = os.listdir(test_dir)
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split(".")[0]
        class_num_str = int(file_str.split("_")[0])
        vector_under_test = img2vector(f"{test_dir}/{file_name_str}")
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print(
            f"the classifier came back with: {classifier_result}, "
            f"the real answer is: {class_num_str}"
        )
        if classifier_result != class_num_str:
            error_count += 1.0
    print(f"the total number of errors is: {error_count}")
    print(f"the total error rate is: {error_count / float(m_test)}")
