import pandas as pd
import seaborn as sns

from pathlib import Path
from IPython.core.display_functions import display

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


class Plots:
    def __init__(self, ref_of_json_file: str, folder_to_save_pictures: str):
        self.__df: pd.DataFrame = pd.read_json(ref_of_json_file)
        self.__paths: dict = {'confusion_matrix': Path(), 'amount_rooms': Path()}
        self.__folder_to_save_pictures: str = folder_to_save_pictures
        self.__path_folder_to_save_pictures: Path = Path(folder_to_save_pictures)

        for key in self.__paths.keys():
            path = self.__path_folder_to_save_pictures / f'{key}.png'
            self.__paths[key] = path

    def display_df(self) -> None:
        display(self.__df)

    def get_paths(self) -> dict:
        return self.__paths.copy()

    def draw_conf_mat_plot(self, need_to_save_file: bool) -> None:
        conf_mat_df = pd.DataFrame(columns=self.__df['rb_corners'].unique(),
                                   data=confusion_matrix(self.__df['rb_corners'], self.__df['gt_corners'],
                                                         normalize='all'),
                                   index=self.__df['rb_corners'].unique())

        sns.heatmap(conf_mat_df, cmap='coolwarm', annot=True)
        plt.title('Looks like over-fitting is here. All values are right :(')
        plt.xlabel('Predicted')
        plt.ylabel('Real')
        if need_to_save_file:
            plt.savefig(self.__paths['confusion_matrix'])
        plt.show()

    def draw_amount_room_plot(self, need_to_save_file: bool) -> None:
        plt.figure(figsize=(10, 8))
        sns.barplot(self.__df['name'].value_counts()[:20], orient='y')
        plt.title('Amount rooms in every type of room')
        if need_to_save_file:
            plt.savefig(self.__paths['amount_rooms'])
        plt.show()

    def draw_distribution_plots(self) -> None:
        for col_name in self.__df.columns[1:]:
            fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(7, 3), height_ratios=[0.8, 0.2], sharex=True)
            sns.histplot(data=self.__df, x=col_name, ax=ax_hist)
            sns.boxplot(data=self.__df, x=col_name, orient="h", ax=ax_box)
            plt.show()

    def draw_mean_and_corr_plot(self) -> None:
        sns.barplot(data=self.__df, x='rb_corners', y='mean', estimator='mean')
        plt.show()

        print(f"correlation: {self.__df['rb_corners'].corr(self.__df['mean'])}")

    def draw_plots(self) -> None:
        self.draw_conf_mat_plot(True)
        print("_" * 100)
        self.draw_amount_room_plot(True)
        print("_" * 100)
        self.draw_distribution_plots()
        print("_" * 100)
        self.draw_mean_and_corr_plot()
