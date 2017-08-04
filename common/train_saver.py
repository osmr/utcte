from utct.common.saver import Saver


class TrainSaver(Saver):

    def __init__(self,
                 work_dir,
                 project_name,
                 model_filename_prefix,
                 data_source,
                 task_name,
                 suffix=""):
        super(TrainSaver, self).__init__(
            work_dirname=work_dir,
            project_name=project_name,
            model_filename_prefix=model_filename_prefix,
            score_log_filename=task_name+"_score.log",
            hyper_log_filename=task_name+"_hyper.log",
            score_log_subdir_name=None,
            score_ref_filename="score_ref.csv")
        data_source.update_project_dirname(self.project_dirname)
