import torch
from torchmetrics import PeakSignalNoiseRatio as _PSNR, StructuralSimilarityIndexMeasure as _SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as _FID


class PSNR(_PSNR):
    def update(self, preds, target):
        super().update(preds.float(), target.float())


class SSIM(_SSIM):
    def update(self, preds, target):
        super().update(preds.float(), target.float())


class FID(_FID):
    def update(self, preds, target):
        super().update(target.expand(-1, 3, -1, -1), real=True)
        super().update(preds.expand(-1, 3, -1, -1), real=False)


if __name__ == "__main__":
    PSNR()
    SSIM()
    FID()
