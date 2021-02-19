import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectorMixin


def get_features_out(estimator, features_in):
    if hasattr(estimator, 'get_feature_names'):
        return estimator.get_feature_names(features_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(features_in)[estimator.get_support()]
    else:
        return features_in


def get_ct_feature_names(ct):
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != 'remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_features_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_features_out(estimator, features)
            output_features.extend(features_out)

        # elif estimator == 'passthrough':
        #     print(name, estimator, features)
            # output_features.extend(ct._feature_names_in(features))

    return output_features
