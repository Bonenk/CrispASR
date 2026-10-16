# Dia F32 reduction-order diagnosis

Use `dispatch.py --help` for a bounded CPU experiment on the accepted original
full-model query/hidden/logit captures and physically verified Q/head weights.
The original pinned `DenseGeneral` runs unchanged in separate MKL CNR modes.
Actual MKL verbose SGEMM records, strides, thread counts and exact repeats are
retained. It never loads a full neural model or changes the generation recipe.

`order.py --help` tests an explicit F32 multiply-then-add recurrence against all
head outputs, calls the actual retained ggml vector dot, and reproduces its
four-by-eight FMA reduction exactly. It then invokes the unchanged pinned
original CFG/filter with the previously verified exponential noise. This is
operator/decision evidence; full unforced generation is still required for a
runtime candidate. Neither a forced token nor a reference-tensor injection is
an acceptable production repair.

Run with the pinned Torch2.6 CPU/NumPy1.26 reference environment; all scratch,
weights and arrays belong on storage. Never reuse an existing output directory.
The calibration explicitly distinguishes the more accurate exact-product dot
from matching the reference's F32 arithmetic. Different CPU arithmetic paths
can be valid approximations to the same real-valued operation and still cross
the measured nucleus-support cutoff.

Intel documents MKL conditional numerical reproducibility and its selected
branches at https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-1/get-started-with-conditional-num-reproducibility.html.
