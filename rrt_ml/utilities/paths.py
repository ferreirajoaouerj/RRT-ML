from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):

    home: Path = Path(__file__).parents[1]

    configs: Path = home / 'configs'
    configs_rl = configs / 'rl'
    configs_rrt = configs / 'rrt'
    configs_sl = configs / 'sl'
    configs_hyper = configs / 'hyper'

    data: Path = home / 'data'
    data_rl: Path = data / 'rl'
    data_rl_distance: Path = data_rl / 'distance'
    data_sl: Path = data / 'sl'
    data_sl_narrow: Path = data_sl / 'narrow'
    data_sl_narrow_train_csv: Path = data_sl_narrow / 'train.csv'
    data_sl_narrow_val_csv: Path = data_sl_narrow / 'val.csv'

    deps: Path = home / 'deps'
    deps_libraries: Path = deps / 'libraries'
    deps_models: Path = deps / 'models'

    environments: Path = home / 'environments'

    experiments: Path = home / 'experiments'
    experiments_rl: Path = experiments / 'rl'
    experiments_sl: Path = experiments / 'sl'
    experiments_rrt: Path = experiments / 'rrt'

    utilities: Path = home / 'utilities'

    def cfg(self, alg: str, name: str, hyper: bool):
        if hyper:
            return self.configs_hyper / (name + '.yaml')
        else:
            match alg:
                case 'rl':
                    path = self.configs_rl / (name + '.yaml')
                case 'sl':
                    path = self.configs_sl / (name + '.yaml')
                case 'rrt':
                    path = self.configs_rrt / (name + '.yaml')
                case _:
                    raise NotImplementedError
            return path

    def exp(self, alg: str, name: str):
        match alg:
            case 'rl':
                path = self.experiments_rl / name
            case 'sl':
                path = self.experiments_sl / name
            case 'rrt':
                path = self.experiments_rrt / name
            case _:
                raise NotImplementedError
        return path

    def exp_fig(self, alg: str, name: str):
        return self.exp(alg=alg, name=name) / 'figs'

    def exp_tensorboard(self, alg: str, name: str):
        match alg:
            case 'rl':  # In MRL tensorboard file is on the parent folder
                return self.exp(alg=alg, name=name)
            case _:
                return self.exp(alg=alg, name=name) / 'tensorboard'

    def exp_checkpoint(self, alg: str, name: str):
        return self.exp(alg=alg, name=name) / 'checkpoint'

    def exp_stats(self, alg: str, name: str):
        return self.exp(alg=alg, name=name) / 'stats'


class PathsIntellisense:
    """
    Visualize all paths and files.
    """

    def __init__(self, *sub_folders):
        """
        Recursively build a list of paths and files
        :param sub_folders: for recursive calls
        """

        home = Path().cwd().parent
        for folder in sub_folders:
            home = home / folder

        self.path = home

        skip_folders = ['.git', '.idea']
        files_and_dirs = [f for f in list(home.glob('*'))]
        for file_or_dir in files_and_dirs:

            name = str(file_or_dir).rsplit("\\")[-1]
            if name in skip_folders:
                continue

            elif not file_or_dir.is_dir():
                name = name.replace('.', '')
                self.__setattr__(name, file_or_dir)

            else:
                call_sub_folders = [f for f in sub_folders]
                call_sub_folders.append(name)
                self.__setattr__(name, PathsIntellisense(*call_sub_folders))

    def __call__(self, *args, **kwargs):
        """
        Call object to get the path of folders.
        :param args: None.
        :param kwargs: None.
        :return: path to folder.
        """

        return self.path


