# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# PyTorch library
#
import torch



class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience = 3, min_delta = 0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                print(f'[INFO] Early stopping')
                return ( True )
            else:
                return ( False )