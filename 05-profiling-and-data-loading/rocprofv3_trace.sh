# rocprofv3_trace.sh
uv run rocprofv3 --sys-trace -- python profile_steps.py      # writes <hostname>/<pid>_results.db
uv run rocpd2pftrace -i "$(ls -t */*_results.db | head -1)"  # writes rocpd-output-data/out_results.pftrace
