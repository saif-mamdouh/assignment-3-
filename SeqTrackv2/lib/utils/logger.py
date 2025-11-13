import time
from datetime import timedelta

class TrainingLogger:
    def __init__(self, log_path, print_every=50):
        self.log_path = log_path
        self.print_every = print_every
        self.epoch_start_time = None
        self.last_log_time = None
        self.samples_seen = 0
        self.total_samples = 0
        open(self.log_path, "w").close()  # clear file

    def start_epoch(self, epoch, total_samples):
        self.epoch_start_time = time.time()
        self.last_log_time = self.epoch_start_time
        self.samples_seen = 0
        self.total_samples = total_samples
        self._write(f"\nEpoch {epoch} : Starting training ({total_samples} samples)\n")

    def step_samples(self, n):
        self.samples_seen += n
        # Check if it's time to log
        if self.samples_seen % self.print_every < n:
            now = time.time()
            time_last_interval = now - self.last_log_time
            total_elapsed = now - self.epoch_start_time
            # estimate remaining time based on last interval
            samples_left = max(self.total_samples - self.samples_seen, 0)
            est_time_left = (time_last_interval / self.print_every) * samples_left

            def fmt(sec):
                return str(timedelta(seconds=int(sec)))

            self._write(
                f"Epoch {self._current_epoch()} : {self.samples_seen} / {self.total_samples} samples ,\n"
                f"time for last {self.print_every} samples : {fmt(time_last_interval)} ,\n"
                f"time since beginning : {fmt(total_elapsed)} ,\n"
                f"time left to finish the epoch : {fmt(est_time_left)}\n"
                f"\n"
            )
            self.last_log_time = now

    def log_metrics(self, epoch, step, **kwargs):
        msg = f"[Epoch {epoch}][Step {step}] "
        msg += " ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
        self._write(msg + "\n")

    def _write(self, msg):
        print(msg.strip())
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _current_epoch(self):
        # Extract current epoch from the last message in log if needed
        return getattr(self, 'current_epoch', 1)
