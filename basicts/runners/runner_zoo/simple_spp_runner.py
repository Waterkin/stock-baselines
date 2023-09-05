import torch

from ..base_spp_runner import BaseStockPricePredictionRunner


class SimpleStockPricePredictionRunner(BaseStockPricePredictionRunner):
    """Simple Runner: select forward features and target features. This runner can cover most cases."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        
    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess
        label, history_data = data
        history_data = self.to_running_device(history_data)      # B, L, 1, C
        label = self.to_running_device(label)       # B, 1, 1, 1
        batch_size, length, features = history_data.shape
        history_data = history_data.reshape(batch_size, length, 1, features)
        label = label.reshape(batch_size, 1, 1, 1) # keep same shape with pred

        # curriculum learning
        prediction = self.model(history_data=history_data, future_data=label, batch_seen=iter_num, epoch=epoch, train=train)
        # feed forward
        assert list(prediction.shape) == [batch_size, 1, 1, 1], \
            f"error shape of the output {prediction.shape}, edit the forward function to reshape it to [B, 1, 1, 1]"
        return prediction, label
    
    