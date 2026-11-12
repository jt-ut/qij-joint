#!/bin/bash
# Job files for Vista's gg partition (144 cores per node) that write the canonical results tree.
#
#   runs/<dataset>_N<N>/{oracle, ij_<estimator>, boot_B<B>, qij_<estimator>_eps<eps>}/
#
# One job = one case and one method, over draws FIRST..LAST (default 0..1000), one draw per core: the draws
# are fanned out with xargs -P 144, each as its own `scripts/run.py ... --workers 1` process, into
# $SCRATCH/QIJ_JOINT/runs (the tree); when the job ends it rsyncs the tree to $WORK/projects/QIJ_JOINT/runs.
# For qij the job takes a list of eps and runs them one after another (each over all its draws). The code is
# the clone at CLONE, staged to node-local /tmp per job (with its .git, so config.json records the commit);
# the job refuses to run if the clone is not at COMMIT. Nothing here submits: each job set gets a
# submit_all.sh that submits in order and stops at the queue's limit (20 running, 40 submitted on gg).
#
#   COMMIT=<12-char hash> ./make_jobs_vista.sh smoke        a 2-draw run of every case and method, 30 min, into
#                                                           $SCRATCH/QIJ_JOINT/smoke (not the tree)
#   COMMIT=<12-char hash> ./make_jobs_vista.sh refs         oracle, ij, boot B5000 for mvt, fp, imf, pareto at N = 10^4
#   COMMIT=<12-char hash> ./make_jobs_vista.sh qij          qij for cloudfil_u p1 (one job per eps) and for the four
#                                                           paper datasets (one job per dataset, all eps)
#
# Written 2026-11-12 for the talk update (qij_joint/spec/PLAN_talk_update.md section 9). Plain bash, no job is
# submitted by this script.
set -euo pipefail

SET=${1:?job set: smoke | refs | qij}
COMMIT=${COMMIT:?COMMIT=<hash the clone must be at>}
CLONE=${CLONE:-/work2/08772/jjt277/stampede3/software/qij_joint}
VENV=${VENV:-/home1/08772/jjt277/envs/qij_vista}
ACCOUNT=AST24021; PARTITION=gg; CORES=144; N=${N:-10000}; SEED=0
FIRST=${FIRST:-0}; LAST=${LAST:-1000}
SCR=${SCRATCH:?SCRATCH not set}/QIJ_JOINT
PROJ=${WORK:?WORK not set}/projects/QIJ_JOINT
JOBS=$PROJ/jobs/$SET; LOGS=$SCR/logs/$SET
EPS_ALL=${EPS_ALL:-"0.01 0.01414213562373095 0.02 0.022360679774997897 0.025819888974716113 0.03162277660168379 0.044721359549995794 0.06324555320336758 0.1 0.1414213562373095 0.2"}

HEAD=$(cd "$CLONE" && git rev-parse --short=12 HEAD)
[ "$HEAD" = "$COMMIT" ] || { echo "REFUSING: $CLONE is at $HEAD, not $COMMIT"; exit 1; }
mkdir -p "$JOBS" "$LOGS"
: > "$JOBS/batches.txt"

# job NAME WALL RUNS_DIR "<one run.py command line per element, without the python and the script path>"...
job() {
    local name=$1 wall=$2 runs=$3; shift 3
    local body=""
    for line in "$@"; do
        # a line is "<dataset> <estimator> <method> [extra flags]"; the draws are fanned out per core
        body+="
echo \"--- \$(date '+%F %T') $line draws $FIRST:$LAST\"
mkdir -p $LOGS/${name}_draws
seq $FIRST $((LAST - 1)) | xargs -P $CORES -I{} sh -c \\
    'python \$STAGE/scripts/run.py $line --N $N --draws \$1:\$((\$1 + 1)) --seed $SEED --workers 1 --out $runs > $LOGS/${name}_draws/\$(echo \"$line\" | tr \" \" _)_s\$1.log 2>&1 || echo \"draw \$1 of [$line] failed (exit \$?)\"' _ {}"
    done
    cat > "$JOBS/$name.sh" <<EOF
#!/bin/bash
#SBATCH -J $name
#SBATCH -o $JOBS/${name}_%j.out
#SBATCH -e $JOBS/${name}_%j.err
#SBATCH -p $PARTITION
#SBATCH -N 1
#SBATCH -n $CORES
#SBATCH -t $wall
#SBATCH -A $ACCOUNT
# Written by make_jobs_vista.sh ($SET) at commit $COMMIT. One draw per core; outputs to $runs.
source $VENV/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=\${SCRATCH:-/tmp}/mplcache
export STAGE=/tmp/qij_stage_\$SLURM_JOB_ID
rsync -a --exclude runs --exclude results $CLONE/ \$STAGE/
export PYTHONPATH=\$STAGE/src
python -c "import qij_joint, sys; f = qij_joint.__file__; print('qij_joint from', f); sys.exit(0 if f.startswith('\$STAGE') else 1)" || { echo "REFUSING: qij_joint not imported from the staged copy"; exit 1; }
[ "\$(git -C \$STAGE rev-parse --short=12 HEAD)" = "$COMMIT" ] || { echo "REFUSING: staged code is not at $COMMIT"; exit 1; }
mkdir -p $runs
echo "=== \$(date '+%F %T') $name at $COMMIT on \$(hostname), $CORES draws at a time"
$body
echo "=== runs done \$(date '+%F %T'); syncing to $PROJ/runs"
mkdir -p $PROJ/runs && rsync -a $runs/ $PROJ/runs/ && echo "=== synced \$(date '+%F %T')"
rm -rf \$STAGE
EOF
    chmod +x "$JOBS/$name.sh"; echo "$name" >> "$JOBS/batches.txt"
}

case $SET in
smoke)
    # two draws of everything, cheap settings, into the smoke tree; the whole set in one job
    FIRST=0; LAST=2
    lines=()
    for c in "cloudfil_u p1" "mvt all" "fp all" "imf all" "pareto all"; do
        lines+=("$c oracle" "$c ij" "$c boot --B 50" "$c qij --eps 0.1")
    done
    job smoke 00:30:00 "$SCR/smoke/runs" "${lines[@]}"
    ;;
refs)
    for ds in mvt fp imf pareto; do
        job "ref_${ds}_oracle_ij" 01:00:00 "$SCR/runs" "$ds all oracle" "$ds all ij"
        wall=02:00:00; [ "$ds" = imf ] && wall=06:00:00      # imf's refit is an optimizer: 5e6 refits per 1000 draws
        job "ref_${ds}_boot" "$wall" "$SCR/runs" "$ds all boot --B 5000"
    done
    ;;
qij)
    for eps in $EPS_ALL; do
        job "qij_cloudfil_u_p1_eps${eps:0:6}" 03:00:00 "$SCR/runs" "cloudfil_u p1 qij --eps $eps"
    done
    for ds in mvt fp imf pareto; do
        lines=(); for eps in $EPS_ALL; do lines+=("$ds all qij --eps $eps"); done
        job "qij_${ds}_all" 04:00:00 "$SCR/runs" "${lines[@]}"
    done
    ;;
*) echo "unknown job set $SET"; exit 1 ;;
esac

cat > "$JOBS/submit_all.sh" <<'EOF'
#!/bin/bash
# Submits every batch in batches.txt not yet in submitted.txt, in order; stops at the first refusal (the
# queue's submit limit). Rerun to continue. Finished draws are skipped by run.py, so a resubmission fills in.
cd "$(dirname "$0")"; touch submitted.txt
while read -r name; do
    grep -qx "$name" submitted.txt && continue
    if out=$(sbatch "$name.sh" 2>&1); then echo "$name  $(echo "$out" | tail -1)"; echo "$name" >> submitted.txt
    else echo "stopped at $name: $(echo "$out" | tail -1)"; exit 0; fi
done < batches.txt
echo "all batches submitted"
EOF
chmod +x "$JOBS/submit_all.sh"
echo "wrote $(wc -l < "$JOBS/batches.txt") job(s) in $JOBS; submit with $JOBS/submit_all.sh"
