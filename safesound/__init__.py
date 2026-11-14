"""SafeSound Tiny: simulation, models and evaluation for safety-sound detection."""
import os

# Many CPU worker processes each run numpy/scipy; one BLAS thread per process avoids
# thousands of threads fighting over the cores. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
