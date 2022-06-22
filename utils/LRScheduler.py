# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# PyTorch library
#
import torch


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5, verbose = True):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience  = patience
        self.min_lr    = min_lr
        self.factor    = factor
        self.verbose   = verbose
        
#         self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer, 
#                                                                        steps     = self.patience, 
#                                                                        verbose   = self.verbose )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode      = 'min',
                patience  = self.patience,
                factor    = self.factor,
                min_lr    = self.min_lr,
                verbose   = self.verbose 
            )
        
    def __call__(self, val_loss):
        self.lr_scheduler.step( val_loss )