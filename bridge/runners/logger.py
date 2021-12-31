from pytorch_lightning.loggers import CSVLogger, NeptuneLogger, TensorBoardLogger, WandbLogger


class Logger:
    def log_metrics(self, metric_dict, step, save=False):
        pass

    def log_hparams(self, hparams_dict):
        pass
