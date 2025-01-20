import numpy as np
import numbers

class PerformanceMetrics:
    def __validate_scoring_inputs(expected, predicted):
        assert isinstance(expected, np.ndarray)
        
        if isinstance(predicted, numbers.Number):
            return 

        if isinstance(predicted, np.ndarray):
            assert len(expected) == len(predicted)

    def MAE(expected, predicted):
        PerformanceMetrics.__validate_scoring_inputs(expected, predicted)
        return np.mean(np.abs(expected - predicted))

    def MSE(expected, predicted):
        PerformanceMetrics.__validate_scoring_inputs(expected, predicted)
        return np.mean((expected - predicted) ** 2)

    def RMSE(expected, predicted):
        PerformanceMetrics.__validate_scoring_inputs(expected, predicted)
        return np.sqrt(np.mean((expected - predicted) ** 2))

    def MAPE(expected, predicted):
        PerformanceMetrics.__validate_scoring_inputs(expected, predicted)
        return np.mean(np.divide(np.abs(expected - predicted), expected))

    def R2(expected, predicted):
        PerformanceMetrics.__validate_scoring_inputs(expected, predicted)
        data_mean = expected.mean()
        residual_sum = np.sum((expected - predicted) ** 2)
        total_sum = np.sum((expected - data_mean) ** 2)
        return 1 - residual_sum / total_sum

    def NRMSE(expected, predicted):
        PerformanceMetrics.__validate_scoring_inputs(expected, predicted)
        rmse = np.sqrt(np.mean((expected - predicted) ** 2))
        return rmse / expected.mean()
