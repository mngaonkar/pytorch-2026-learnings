# tracelens_report.sh
TRACE=$(ls -t log/*.pt.trace.json | head -1)          # newest trace from profile_steps.py
uvx --from git+https://github.com/AMD-AGI/TraceLens.git \
  TraceLens_generate_perf_report_pytorch --profile_json_path "$TRACE"
