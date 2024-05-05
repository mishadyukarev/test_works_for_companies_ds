import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class EDAdf:
    def __init__(self, name_train_df: str, num_cols: list[str]):
        self.__df = pd.read_csv(name_train_df)
        self.__num_cols = num_cols
        self.__cat_cols = [col for col in self.__df.columns if col not in num_cols]

    def get_df_copy(self):
        return self.__df.copy()

    def display(self):
        display(self.__df)

    def show_missing_values(self):
        plt.figure(figsize=(10, 5))
        plt.title("Missing values")
        sns.barplot(self.__df.isnull().sum(), orient='h')

    def visualize_box_and_hist_plots(self):
        for reg_col in self.__num_cols:
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
            sns.boxplot(self.__df[reg_col], orient="h", ax=ax_box)
            sns.histplot(x=self.__df[reg_col], ax=ax_hist)
            ax_box.set(xlabel='')
            plt.show()

    def visualize_bar_plots(self):
        for cat_col in self.__cat_cols:
            sns.barplot(self.__df[cat_col].value_counts())
            plt.show()
