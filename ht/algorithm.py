import numpy as np
from tqdm import tqdm


class GraphMatrixModelAlgorithm:
    def __init__(self, adjacency_matrix, items_range):
        markov_matrix = np.copy(adjacency_matrix)
        self.dim = len(markov_matrix)

        # normalize
        for i in range(self.dim):
            norm = np.sum(markov_matrix[i])
            if norm != 0:
                markov_matrix[i] = np.where(markov_matrix[i] != 0, 0.99 / norm, 0.01 / (self.dim - norm))
            else:
                markov_matrix[i] = np.ones_like(markov_matrix[i]) / self.dim

        self.markov_matrix = markov_matrix
        self.items_range = items_range


class HittingTimeAlgorithm(GraphMatrixModelAlgorithm):
    def __init__(self, model):
        """
        Parameters initialization of hitting time algorithm
        :param model: model
        :type model GraphMatrixModel
        """
        super().__init__(model.adjacency_matrix, model.item_entry_range)

        # compute stationary probabilities
        sum_weighted = np.sum(model.adjacency_matrix)
        pi = np.sum(model.adjacency_matrix, axis=1) / sum_weighted
        self.pi = pi

    def run(self, entries, excluded_dict=None, k=10, time_steps=5):
        """
        Run the hitting time algorithm with partial update
        :param entries: target user ids
        :param excluded_dict: excluded item ids when recommending
        :type excluded_dict dict
        :param time_steps: iteration steps
        :param k: top-k items to be recommended
        :return: recommendations
        """
        markov_matrix = self.markov_matrix

        if excluded_dict is None:
            excluded_dict = {}
        if k > self.dim:
            k = round(pow(self.dim, 0.25))
            if k == 0:
                k = 1
        rec_start, rec_end = self.items_range

        pi = self.pi

        for _ in range(time_steps):
            markov_matrix = np.matmul(markov_matrix, markov_matrix)  # update matrix

        sorted_rec_ids = []
        for entry in entries:
            # compute hitting time
            hitting_times = pi[rec_start:rec_end] / (pi[entry] * markov_matrix[entry][rec_start:rec_end])

            sorted_rec_id = np.argsort(hitting_times)
            excluded_list = []
            if entry in excluded_dict:
                excluded_list = excluded_dict[entry]
            if len(excluded_list) > 0:
                sorted_rec_id = np.array([rec_id for rec_id in sorted_rec_id if rec_id not in excluded_list])
            sorted_rec_ids.append(list(sorted_rec_id[0:min(k, len(sorted_rec_id))]))

        return sorted_rec_ids


class AbsorbingTimeAlgorithm(GraphMatrixModelAlgorithm):
    def __init__(self, model):
        """
        Parameters initialization of hitting time algorithm
        :param model: model
        :type model GraphMatrixModel
        """
        super().__init__(model.adjacency_matrix, model.item_entry_range)

    def markov_proceed(self, time_steps=5):
        """
        Proceed the markov process
        :param time_steps: iteration steps
        """
        for _ in tqdm(range(time_steps)):
            self.markov_matrix = np.matmul(self.markov_matrix, self.markov_matrix)  # update matrix
        return self

    def run(self, excluded_list=None, k=10, tau=15):
        """
        Run the hitting time algorithm with partial update
        :param excluded_list: excluded item ids when recommending
        :type excluded_list []
        :param k: top-k items to be recommended
        :param tau: tau iterations of AT
        :return: recommendations
        """
        if excluded_list is None:
            excluded_list = []
        if k > self.dim:
            k = round(pow(self.dim, 0.25))
            if k == 0:
                k = 1
        rec_start, rec_end = self.items_range

        markov_matrix = self.markov_matrix[rec_start:rec_end, rec_start:rec_end]
        at_vec = np.zeros(rec_end - rec_start)
        mask_vec = np.ones(rec_end - rec_start)  # mask already rated items
        for i in excluded_list:
            mask_vec[i] = 0

        for _ in tqdm(range(tau)):
            at_vec = np.add(1, np.matmul(at_vec, markov_matrix)) * mask_vec

        sorted_rec_id = np.argsort(at_vec)
        sorted_rec_id = np.array([rec_id for rec_id in sorted_rec_id if rec_id not in excluded_list])

        return sorted_rec_id[0:min(k, len(sorted_rec_id))]


class AbsorbingCostAlgorithm(GraphMatrixModelAlgorithm):
    def __init__(self, model):
        """
        Parameters initialization of hitting time algorithm
        :param model: model
        :type model GraphMatrixModel
        """
        super().__init__(model.adjacency_matrix, model.item_entry_range)

    def markov_proceed(self, time_steps=5):
        """
        Proceed the markov process
        :param time_steps: iteration steps
        """
        for _ in tqdm(range(time_steps)):
            self.markov_matrix = np.matmul(self.markov_matrix, self.markov_matrix)  # update matrix
        return self

    def run(self, entropy, excluded_list=None, k=10, tau=15):
        """
        Run the hitting time algorithm with partial update
        :param entropy: user entropy vector
        :param excluded_list: excluded item ids when recommending
        :type excluded_list []
        :param k: top-k items to be recommended
        :param tau: tau iterations of AT
        :return: recommendations
        """
        if excluded_list is None:
            excluded_list = []
        if k > self.dim:
            k = round(pow(self.dim, 0.25))
            if k == 0:
                k = 1
        rec_start, rec_end = self.items_range

        markov_matrix = self.markov_matrix[rec_start:rec_end, rec_start:rec_end]
        at_vec = np.zeros(rec_end - rec_start)
        mask_vec = np.ones(rec_end - rec_start)  # mask already rated items
        for i in excluded_list:
            mask_vec[i] = 0

        for _ in tqdm(range(tau)):
            at_vec = np.add(np.matmul(entropy, markov_matrix), np.matmul(at_vec, markov_matrix)) * mask_vec

        sorted_rec_id = np.argsort(at_vec)
        sorted_rec_id = np.array([rec_id for rec_id in sorted_rec_id if rec_id not in excluded_list])

        return sorted_rec_id[0:min(k, len(sorted_rec_id))]
