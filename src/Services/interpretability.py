import matplotlib
from Services.utils import get_ct_feature_names


class InterpretabilityService:
    def __init__(self, clf) -> None:
        self.clf = clf

    def permutation_importance(self, X_valid, y_valid):
        import eli5
        from eli5.sklearn import PermutationImportance

        model = self.clf.named_steps['model']
        preprocessor = self.clf.named_steps['preprocessor']
        feature_names = ['Id']
        feature_names.extend(get_ct_feature_names(preprocessor))
        X_valid = preprocessor.transform(X_valid).toarray()

        perm = PermutationImportance(
            model, random_state=42).fit(X_valid, y_valid)
        self.perm_weights = eli5.explain_weights(
            perm, feature_names=feature_names)
        print(eli5.format_as_text(self.perm_weights))

    def partial_plot(self, X_valid, feature_name):
        from pdpbox import pdp
        pdp_feature = pdp.pdp_isolate(
            model=self.clf, dataset=X_valid, model_features=X_valid.columns, feature=feature_name)
        pdp.pdp_plot(pdp_feature, feature_name,
                     plot_pts_dist=True, plot_lines=True)
        pdp.plt.savefig(f'outputs/pdp_{feature_name}.png', dpi=200)

    def partial_plot_2D(self, X_valid, features_to_plot):
        from pdpbox import pdp

        interact = pdp.pdp_interact(model=self.clf, dataset=X_valid,
                                    model_features=X_valid.columns, features=features_to_plot)

        try:
            pdp.pdp_interact_plot(
                pdp_interact_out=interact, feature_names=features_to_plot, plot_type='contour')
            pdp.plt.savefig(
                f'outputs/2D_pdp_{features_to_plot[0]}_{features_to_plot[1]}', dpi=200)
        except TypeError:
            import os
            file_to_edit = os.environ['CONDA_PREFIX'] + \
                "/lib/python3.6/site-packages/pdpbox/pdp_plot_utils.py"
            print(f"\nYou need to edit your file {file_to_edit}",
                  "In order to make your package pdpbox compatible with matplotlib 3.x, edit this line :",
                  "	inter_ax.clabel(c2, contour_label_fontsize=fontsize, inline=1) \n",
                  "Replace 'contour_label_fontsize=fontsize' by 'fontsize=fontsize'",
                  "You can run this command :",
                  f"	sed -i 's/contour_label_fontsize/fontsize/g' {file_to_edit}", sep='\n')

    def shap_values(self, X_train, X_valid):
        import shap

        feature_names = X_train.columns

        def model_predict(data_asarray):
            import pandas as pd
            data_asframe = pd.DataFrame(data_asarray, columns=feature_names)
            return self.clf.predict(data_asframe)

        X_train_summary = shap.sample(X_train, 10)
        k_explainer = shap.KernelExplainer(model_predict, X_train_summary)
        k_shap_values = k_explainer.shap_values(X_valid.iloc[0, :])
        print(k_shap_values)
        shap.force_plot(k_explainer.expected_value, k_shap_values, X_valid.iloc[0, :],
                        show=False, matplotlib=True).savefig('outputs/k_shap_values.png', dpi=200)

    def summary_plot(self, X_train, X_valid):
        import shap
        import matplotlib.pyplot as plt

        feature_names = X_train.columns

        def model_predict(data_asarray):
            import pandas as pd
            data_asframe = pd.DataFrame(data_asarray, columns=feature_names)
            return self.clf.predict(data_asframe)

        X_train_summary = shap.sample(X_train, 10)
        explainer = shap.KernelExplainer(model_predict, X_train_summary)
        k_shap_values = explainer.shap_values(X_valid)
        shap.summary_plot(k_shap_values, X_valid, show=False)
        plt.tight_layout()
        plt.savefig('outputs/k_summary_plot.png', dpi=200)
