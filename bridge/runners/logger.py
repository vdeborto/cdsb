from pytorch_lightning.loggers import CSVLogger as _CSVLogger, NeptuneLogger, TensorBoardLogger, WandbLogger as _WandbLogger


class Logger:
    def log_metrics(self, metric_dict, step=None):
        pass

    def log_hyperparams(self, params):
        pass

    def log_image(self, key, images, **kwargs):
        pass


class CSVLogger(_CSVLogger):
    def log_image(self, key, images, **kwargs):
        pass


class WandbLogger(_WandbLogger):
    LOGGER_JOIN_CHAR = '/'

    def log_metrics(self, metrics, step=None):
        fb = metrics.pop('fb')
        metrics = {fb + '/' + k: v for k, v in metrics.items()}
        super().log_metrics(metrics, step=step)
