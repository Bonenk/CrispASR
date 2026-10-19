# Long-input CUDA alignment validation (issue550)

Use the external maintainer Kaggle guide. This worker consumes existing GitHub
Actions artifacts and runs inference only; it never compiles native code or
loads Torch. Before pushing, check account quota and concurrent sessions. Push
with `tools/kaggle/kpush.py`; account and credentials must remain external.

The script pins source191815511, the CUDA12.4.1 sm60/sm75 bundle from
run39215806310, and the accepted CPU fixture archive from39212307554. Both
Actions ZIPs, the native archive, every packaged runtime member, model and
fixture inputs are verified. The attached credential dataset needs Actions
artifact read access and the ordinary HF token. Expired artifacts require a
reviewed replacement and new pins; never silently fetch the latest build.

The public Python/C ABI cache is reused across44.52s,2703s and44.52s of exactly
repeated licensed Japanese FLEURS speech. The last short call must reproduce the
first with an exact12.34s absolute offset. Each call must retain every native
unit with positive ordered bounds under the existing one-frame allowance. A
fresh CLI call must agree with the first public result. Native graph assignment
logs must show CUDA splits for both public and CLI routes, with observed device
utilization during the long call and the packaged CUDA libraries actually
mapped in the public process. Device memory/utilization samples include an idle
baseline; reported peaks are sampled device totals, not allocator maxima.

Retain `validation.json`, `*-before.json`, `long.json`, `short-after.json`, CLI
output/log, public native log, GPU samples and attribution. Failure returns a
nonzero status and preserves partial evidence. Review raw artifacts before
publishing a portable audit or closing550. This does not prove natural45-minute
speech quality, ground-truth word boundaries, RTX3090-specific behavior,
CPU/GPU byte parity or constant host RAM. Host logits and traceback grow with
recording length; align-only does not apply VAD.
