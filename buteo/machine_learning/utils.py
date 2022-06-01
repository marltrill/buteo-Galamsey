import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from numba import jit, prange

def tpe(y_true, y_pred):
    epsilon = 1e-7
    pred_sum = tf.math.reduce_sum(y_pred)
    true_sum = tf.math.reduce_sum(y_true)
    ratio = tf.math.divide(pred_sum, true_sum + epsilon)

    return ratio

def mse_mae_mix_loss(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return tf.math.multiply(mse, mae)

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric="val_loss", this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float("-inf")
        else:
            self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        metric_value = abs(logs[self.save_best_metric])
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor=["loss", "val_loss"]):
        self.time_started = None
        self.time_finished = None
        self.monitor = monitor
        
    def on_train_begin(self, logs=None):
        self.time_started = datetime.now()
        print(f'\nTraining started: {self.time_started}\n')
        
    def on_train_end(self, logs=None):
        self.time_finished = datetime.now()
        train_duration = str(self.time_finished - self.time_started)
        print(f'\nTraining finished: {self.time_finished}, duration: {train_duration}')
        
        metrics = [] 
        for metric in self.monitor:
            str_val = str(logs[metric])
            before_dot = len(str_val.split(".")[0])

            spaces = 16 - (len(metric) + before_dot)
            if spaces <= 0:
                spaces = 1

            pstr = f"{metric}:{' ' * spaces}{logs[metric]:.4f}"
            metrics.append(pstr)

        print('\n'.join(metrics))

class OverfitProtection(tf.keras.callbacks.Callback):
    def __init__(self, difference=0.1, patience=3, offset_start=3, verbose=True):
        self.difference = difference
        self.patience = patience
        self.offset_start = offset_start
        self.verbose = verbose
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        
        if epoch < self.offset_start:
            return

        epsilon = 1e-7
        ratio = loss / (val_loss + epsilon)

        if (1.0 - ratio) > self.difference:
            self.count += 1

            if self.verbose:
                print(f"Overfitting.. Patience: {self.count}/{self.patience}")

        elif self.count != 0:
            self.count -= 1
        
        if self.count >= 3:
            self.model.stop_training = True

            if self.verbose:
                print(f"Training stopped to prevent overfitting. Difference: {ratio}, Patience: {self.count}/{self.patience}")


class LearningRateAdjuster(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch, decay_rate=0.95, decay_rate_epoch=10, set_at_end="end_rate", step_wise=True, verbose=True):
        self.start_epoch = start_epoch
        self.decay_rate = decay_rate
        self.decay_rate_epoch = decay_rate_epoch
        self.set_at_end = set_at_end
        self.step_wise = step_wise
        self.initial_epoch = 0
        self.decay_count = 0
        self.verbose = verbose
        self.initial_lr = None

    def on_epoch_end(self, epoch, logs=None):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * self.decay_rate ^ ((epoch - self.start_epoch) / self.decay_rate_epoch)

        if self.step_wise and epoch == (self.start_epoch + (self.decay_rate_epoch * (self.decay_count + 1))):
            if self.verbose:
                print(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")

            self.model.optimizer.lr.assign(new_lr)
            self.decay_count += 1

        elif not self.step_wise:
            if self.verbose:
                print(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")

            self.model.optimizer.lr.assign(new_lr)
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.initial_lr is None:
            self.initial_lr = self.model.optimizer.lr.read_value()

    def on_train_end(self, logs=None):
        if self.set_at_end == "end_rate":
            pass
        elif isinstance(self.set_at_end, (int, float)):
            self.model.optimizer.lr.assign(self.set_at_end)
        elif self.set_at_end == "initial":
            self.model.optimizer.lr.assign(self.initial_lr)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def interp_array(
    arr,
    min_vals,
    max_vals,
    min_vals_adj,
    max_vals_adj,
):
    out_arr = np.empty_like(arr)
    for img in prange(arr.shape[0]):
        for band in range(arr.shape[3]):
            min_val = min_vals[img, 0, 0, band]
            min_val_adj = min_vals_adj[img, 0, 0, band]

            max_val = max_vals[img, 0, 0, band]
            max_val_adj = max_vals_adj[img, 0, 0, band]

            out_arr[img, :, :, band] = np.interp(
                arr[img, :, :, band], (min_val, max_val), (min_val_adj, max_val_adj)
            )

    return out_arr


def image_augmentation(list_of_inputs, in_place=True, options=None):
    """
    Augment the input images with random flips, and noise.
    """
    if not isinstance(list_of_inputs, list):
        list_of_inputs = [list_of_inputs]

    base_options = {
        "scale": 0.1,
        "shift": 0.1,
        "band": 0.05,
        "contrast": 0.05,
        "pixel": 0.025,
        "clamp": True,
        "clamp_max": 1,
        "clamp_min": 0,
    }

    if options is None:
        options = base_options
    else:
        for key in options:
            if key not in base_options:
                raise ValueError(f"Invalid option: {key}")
            base_options[key] = options[key]
        options = base_options

    x_outputs = []
    for idx, arr in enumerate(list_of_inputs):
        if in_place:
            base = np.array(arr, copy=True, dtype="float32")
        else:
            base = list_of_inputs[idx]

        scale = np.random.normal(1.0, options["scale"], (len(base), 1, 1, 1))
        shift = np.random.normal(1.0, options["shift"], (len(base), 1, 1, 1))

        cmax = np.random.normal(1.0, options["contrast"], (base.shape[0], 1, 1, 1))
        cmin = np.random.normal(1.0, options["contrast"], (base.shape[0], 1, 1, 1))

        band = np.random.normal(
            1.0, options["band"], (base.shape[0], 1, 1, base.shape[3])
        )

        pixel = np.random.normal(1.0, options["pixel"], base.shape)

        if in_place:
            list_of_inputs[idx] = (list_of_inputs[idx] + shift) * scale * band * pixel 
            base = list_of_inputs[idx]
        else:
            base = (base + shift) * scale * band * pixel

        min_vals = base.min(axis=(1, 2)).reshape((base.shape[0], 1, 1, base.shape[3]))
        max_vals = base.max(axis=(1, 2)).reshape((base.shape[0], 1, 1, base.shape[3]))
        min_vals_adj = min_vals * cmin
        max_vals_adj = max_vals * cmax
        min_vals_adj = np.where(min_vals_adj >= max_vals_adj, min_vals, min_vals_adj)
        max_vals_adj = np.where(max_vals_adj <= min_vals_adj, max_vals, max_vals_adj)

        if in_place:
            list_of_inputs[idx] = interp_array(list_of_inputs[idx], min_vals, max_vals, min_vals_adj, max_vals_adj)
            base = list_of_inputs[idx]
        else:
            base = interp_array(base, min_vals, max_vals, min_vals_adj, max_vals_adj)

        if options["clamp"]:
            if in_place:
                list_of_inputs[idx] = np.interp(
                    base,
                    (base.min(), base.max()),
                    (options["clamp_min"], options["clamp_max"]),
                )
                base = list_of_inputs[idx]
            else:
                base = np.interp(
                    base,
                    (base.min(), base.max()),
                    (options["clamp_min"], options["clamp_max"]),
                )

        if not in_place:
            x_outputs.append(base)

    if in_place:
        return list_of_inputs

    return x_outputs
