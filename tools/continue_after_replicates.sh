#!/usr/bin/env bash
# Campaign 2: wait for the running `rct_supervise auto` to finish, judge its
# replicate block against criteria fixed BEFORE seeing it, and if they pass
# start open-ended collection (`rct_supervise run`, no deadline) that runs
# until the user stops it.
#
# Criteria: supervisor not halted and not stopped by the user; >= 8 replicated
# probes; >= 90% of replicate runs valid; collision AND stall outcomes agree
# across all runs of a probe for >= 50% of probes.
# Decision -> campaigns/campaign2/supervisor/continue_decision.json
# Create campaigns/campaign2/supervisor/NO_CONTINUE to veto continuing.
set -u
C=/home/forough/phd_projects/online_tuner/campaigns/campaign2
SRC='source /opt/ros/humble/setup.bash && source /home/forough/phd_projects/online_tuner/install/setup.bash && cd /home/forough/phd_projects/online_tuner'

while docker exec ros2-tiago-dev bash -c 'ps -eo cmd | grep -v grep | grep -q "rct_supervise auto"'; do
  sleep 60
done
echo "auto finished at $(date)"

DECISION=$(docker exec -i ros2-tiago-dev python3 - <<'EOF'
import json, csv, os, datetime
C = "/home/forough/phd_projects/online_tuner/campaigns/campaign2"
st = json.load(open(C + "/supervisor/status.json"))
ev = [json.loads(l) for l in open(C + "/supervisor/events.jsonl") if l.strip()]
last_auto = max([i for i, e in enumerate(ev) if e.get("event") == "auto_start"] or [0])
ev = ev[last_auto:]                                  # this auto run only
kinds = [e.get("event") for e in ev]
card = json.load(open(C + "/dataset/dataset_card.json"))
rep = card.get("replicates") or {}
fails = []
if st.get("phase") == "HALTED" or "halt" in kinds:
    fails.append("supervisor halted: " + str(st.get("halt_reason") or [e for e in ev if e.get("event") == "halt"][-1:]))
if os.path.exists(C + "/supervisor/STOP") or "stopped_by_request" in kinds \
        or os.path.exists(C + "/supervisor/NO_CONTINUE"):
    fails.append("stopped by the user")
if not rep or rep.get("n_probes", 0) < 8:
    fails.append(f"replicate block missing or too small: {rep}")
else:
    p = C + "/dataset/replicates.csv"
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    rr = [r for r in rows if r.get("attempt_kind") == "replicate"]
    v = sum(str(r.get("probe_valid")).strip() in ("1", "1.0", "True") for r in rr) / max(1, len(rr))
    if v < 0.9:
        fails.append(f"replicate runs valid {v:.2f} < 0.90")
    if rep.get("collision_all_agree", 0) < 0.5:
        fails.append(f"collision agreement {rep.get('collision_all_agree')} < 0.5")
    if rep.get("stall_all_agree", 0) < 0.5:
        fails.append(f"stall agreement {rep.get('stall_all_agree')} < 0.5")
dec = {"decided_at": datetime.datetime.now().isoformat(timespec="seconds"),
       "continue": not fails, "fails": fails, "replicates": rep,
       "checkpoint": st.get("checkpoint"),
       "criteria": "valid>=0.9, collision & stall agreement >= 0.5, >=8 probes, no halt/stop"}
json.dump(dec, open(C + "/supervisor/continue_decision.json", "w"), indent=1)
print(json.dumps(dec))
EOF
)
echo "$DECISION"
if echo "$DECISION" | grep -q '"continue": true'; then
  docker exec -d ros2-tiago-dev bash -c "$SRC && exec ros2 run rct_collector rct_supervise run >> campaigns/campaign2/supervisor/run_console.log 2>&1 < /dev/null"
  sleep 120
  echo "--- run started:"; tail -4 $C/supervisor/run_console.log
else
  echo "NOT continuing"
fi
