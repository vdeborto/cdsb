import torch
from accelerate import Accelerator as _Accelerator
from accelerate.state import is_deepspeed_available


if is_deepspeed_available():
    import deepspeed
    from accelerate.deepspeed_utils import DeepSpeedEngineWrapper, DeepSpeedOptimizerWrapper


class Accelerator(_Accelerator):
    def __init__(self, train_batch_size=None, **kwargs):
        super().__init__(**kwargs)
        self.train_batch_size = train_batch_size
        if self.state.deepspeed_plugin is not None:
            deepspeed_plugin = self.state.deepspeed_plugin
            deepspeed_plugin.fp16 = self.state.use_fp16
            deepspeed_plugin.deepspeed_config["train_batch_size"] = self.train_batch_size * deepspeed_plugin.gradient_accumulation_steps
            self.deepspeed_config = deepspeed_plugin.deepspeed_config
            deepspeed_plugin.ds_config = deepspeed_plugin.deepspeed_config

    def _prepare_deepspeed(self, *args):

        deepspeed_plugin = self.state.deepspeed_plugin
        # self.deepspeed_config = deepspeed_plugin.deepspeed_config
        #
        # self.deepspeed_config["train_batch_size"] = (
        #         self.train_batch_size * deepspeed_plugin.gradient_accumulation_steps * self.num_processes
        # )

        result = [self._prepare_one(obj) if isinstance(obj, torch.utils.data.DataLoader) else obj for obj in args]

        model = None
        optimizer = None
        for obj in result:
            if isinstance(obj, torch.nn.Module):
                model = obj
            elif isinstance(obj, (torch.optim.Optimizer, dict)):
                optimizer = obj

        if deepspeed_plugin.auto_opt_mapping:
            is_adam = isinstance(optimizer, torch.optim.Adam)
            is_adamw = isinstance(optimizer, torch.optim.AdamW)
            if (is_adam or is_adamw) and deepspeed_plugin.offload_optimizer_device == "cpu":
                defaults = optimizer.defaults
                optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
                    model.parameters(),
                    lr=defaults["lr"],
                    bias_correction=True,
                    betas=defaults["betas"],
                    eps=defaults["eps"],
                    weight_decay=defaults["weight_decay"],
                    amsgrad=defaults["amsgrad"],
                    adamw_mode=is_adamw,
                )

        # useful when only eval_dataloader is given into `accelerator.prepare()`
        if model is not None:
            engine = DeepSpeedEngineWrapper(
                args=None,
                model=model,
                optimizer=optimizer,
                config_params=self.deepspeed_config,
                dist_init_required=False,
            )
            for i in range(len(result)):
                if isinstance(result[i], torch.nn.Module):
                    result[i] = engine
                elif isinstance(result[i], torch.optim.Optimizer):
                    result[i] = DeepSpeedOptimizerWrapper(engine.optimizer, engine)
            self.deepspeed_engine = engine  # pointing for deepspeed_engine.backward()
            self._models.append(engine)
            self._optimizers.append(engine.optimizer)
            assert (
                    len(self._models) == 1
            ), "You can't use same `Accelerator()` instance with 2 models when using DeepSpeed"

        return tuple(result)

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        """
        Should be used in place of :func:`torch.nn.utils.clip_grad_norm_`.
        """
        self.unscale_gradients()
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
        return total_norm

    def clip_grad_value_(self, parameters, clip_value):
        """
        Should be used in place of :func:`torch.nn.utils.clip_grad_value_`.
        """
        self.unscale_gradients()
        total_norm = torch.nn.utils.clip_grad_value_(parameters, clip_value)
        return total_norm
