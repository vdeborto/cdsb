import deepspeed


class FusedAdam(deepspeed.ops.adam.FusedAdam):
    def zero_grad(self, set_to_none=True):
        super(deepspeed.ops.adam.FusedAdam, self).zero_grad(set_to_none=self.set_grad_none)