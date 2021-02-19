import pandas as pd
from Services.training import TrainingService
from Services.scoring import ScoringService
from Services.interpretability import InterpretabilityService
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    data_train = pd.read_csv('./Data/train.csv')
    data_test = pd.read_csv('./Data/test.csv')
    target_name = 'SalePrice'

    # Train
    trainService = TrainingService(data_train, target_name)
    trainService.train()
    trainService.save_clf()  # Save your classifier

    # Score
    scoreService = ScoringService(trainService.clf)
    print('RMSLE train :', scoreService.score(
        trainService.X_train, trainService.y_train))
    print('RMSLE valid :', scoreService.score(
        trainService.X_valid, trainService.y_valid))
    print('Average RMSLE score (CV):', scoreService.cv_score(
        trainService.X, trainService.y, cv=5))

    y_preds_test = scoreService.score(data_test)
    submission = data_test[['Id']].copy()
    submission['SalePrice'] = y_preds_test
    submission.to_csv('outputs/submission_test.csv',
                      index=False)     # Save your predictions

    # Interpret
    interpService = InterpretabilityService(trainService.clf)
    interpService.permutation_importance(
        trainService.X_valid, trainService.y_valid)
    interpService.partial_plot(trainService.X_valid, 'OverallQual')
    interpService.partial_plot(trainService.X_valid, 'LotArea')
    interpService.partial_plot_2D(
        trainService.X_valid, ['OverallQual', 'LotArea'])
    interpService.shap_values(trainService.X_train, trainService.X_valid)
    interpService.summary_plot(trainService.X_train, trainService.X_valid)
