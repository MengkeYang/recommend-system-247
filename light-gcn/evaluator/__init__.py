try:
    from evaluator.cpp.evaluate_foldout import eval_score_matrix_foldout

    print("eval_score_matrix_foldout with cpp")
except:
    from evaluator.python.evaluate_foldout import eval_score_matrix_foldout

    print("eval_score_matrix_foldout with python")

from evaluator.python.evaluate_foldout import argmax_top_k


def save_ranking(score_matrix, test_items, f_ptr, top_k=50):
    for idx in range(len(test_items)):
        scores = score_matrix[idx]  # all scores of the test user

        ranking = argmax_top_k(scores, top_k)  # Top-K items
        f_ptr.write(','.join(list(map(str, [idx] + list(ranking)))))
        f_ptr.write('\n')
