import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


def cross_validation_process(model, modelX, modelY, method, rangeK):
    meanError = -10000000000000000000000000
    best_k = 0
    estimator_by_k = []
    score_by_k = []
    for i in range(2, rangeK):
        k = KFold(n_splits=i, random_state=1, shuffle=True)
        cv_results = cross_validate(model, modelX, modelY, cv=k, scoring=method, return_estimator=True)
        average_mean_error = np.mean(cv_results['test_score'])
        estimator_cv_results = []
        for model in cv_results['estimator']:
            model.coef_ = np.append(model.coef_, model.intercept_)
            estimator_cv_results.append(model.coef_)
        if average_mean_error > meanError:
            meanError = average_mean_error
            best_k = i
        elif average_mean_error == meanError:
            if i > best_k:
                best_k = i
        estimator_by_k.append(estimator_cv_results)
        score_by_k.append(cv_results['test_score'])
    return best_k, score_by_k, estimator_by_k


def get_coefficient(bestK, scores, estimators):
    best_k_index = bestK - 2
    large_neg_mse = np.max(scores[best_k_index])
    model_index = np.where(scores[best_k_index] == large_neg_mse)[0][0]
    return estimators[best_k_index][model_index]


def predict(predictionX_list, coefficients):
    predict_y = []
    case_num = len(predictionX_list)
    estimator_num = len(predictionX_list.iloc[0])
    for i in range(case_num):
        case = predictionX_list.iloc[i]
        y_bar = coefficients[-1]
        for j in range(estimator_num):
            y_bar += (case[j]*coefficients[j])
        predict_y.append(y_bar)
    return predict_y


