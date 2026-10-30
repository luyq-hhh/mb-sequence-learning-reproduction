"""p01 原规模单案例：默认检查输入，--run 才学习并测试。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
from scipy.io import loadmat

BASE_DIR = Path(__file__).resolve().parent
TIME_SCALE_FACTOR = 10  # 放慢真实执行速度；模型时间步仍为 1 ms。


def check_input(data_file):
    data = loadmat(data_file)
    channels = data["idx_p"][0]
    times = data["t_p"][0].astype(float)
    assert channels.shape == times.shape and len(times) > 0
    assert np.all(np.diff(times) >= 0)
    assert np.all((channels >= 0) & (channels < 840))
    duration = float(times[-1] - times[0])
    info = {"file": data_file.name, "sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
            "events": len(times), "learning_ms": duration * 0.5, "testing_ms": duration}
    print(json.dumps(info), flush=True)
    return info


def check_recording(output_dir, simulation_ms):
    # 沿用现有原始时间戳保存；不以 Neo 数组长度推断完整时间轴。
    with np.load(output_dir / "kc_a_raw_voltage.npz") as raw:
        expected = np.arange(int(simulation_ms), dtype=float)
        complete = (not bool(raw["missing_data"])
                    and np.array_equal(raw["time_ms"], expected)
                    and np.all(np.isfinite(raw["voltage_mv"])))
    from spinn_front_end_common.interface.provenance import ProvenanceReader
    with ProvenanceReader() as reader:
        rows = reader.run_query("""
            SELECT core_name, x, y, processor, description, the_value
            FROM core JOIN core_provenance USING (core_id)
        """)
    with (output_dir / "core_provenance.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["core_name", "x", "y", "p", "counter", "value"])
        writer.writerows(rows)
    bad_rows = [row for row in rows if int(row[5]) > 0 and
                any(word in str(row[4]).lower() for word in
                    ["overload", "overflow", "lost", "dropped", "late", "skipped"])]
    with (output_dir / "activity.pkl").open("rb") as file:
        activity = pickle.load(file)  # 只读取本次自己生成的输出。
    mbon_complete = all(len(activity[name]) == int(simulation_ms)
                        for name in ["mbon_v", "mbon_a_v"])
    checks = {"kc_a_voltage_complete": bool(complete),
              "mbon_voltage_complete": mbon_complete,
              "core_counter_rows": len(rows), "nonzero_error_counters": len(bad_rows)}
    (output_dir / "checks.json").write_text(json.dumps(checks, indent=2))
    assert complete and mbon_complete, "电压记录不完整，保留输出并停止后续阶段"
    assert rows and not bad_rows, "核心运行计数缺失或存在异常，保留输出并停止后续阶段"


def run_case(data_file, output_dir, input_info):
    from importlib.metadata import version
    versions = {name: version(name) for name in ["sPyNNaker", "PyNN", "numpy", "neo"]}
    # 原始电压读取使用 7.4.1 的数据库结构，版本不符时在申请硬件前停止。
    assert versions["sPyNNaker"].split("!")[-1] == "7.4.1", versions
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(BASE_DIR / "src"))
    from MB_main_11_ad_offset import MB_LE, sim, generate_pnlist
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {"status": "RUNNING", "input": input_info, "versions": versions,
              "pn": 840, "kc_per_group": 4000, "time_step_ms": 1,
              "time_scale_factor": TIME_SCALE_FACTOR, "completed_phases": []}
    weights = None
    try:
        for phase in ["learning", "testing"]:
            initialized = False
            phase_dir = output_dir / phase
            try:
                sim.setup(timestep=1.0, min_delay=1.0, max_delay=14,
                          time_scale_factor=TIME_SCALE_FACTOR)
                initialized = True
                sim.set_number_of_neurons_per_core(sim.IF_curr_exp,
                                                   50 if phase == "learning" else 100)
                np.random.seed(1)
                if phase == "learning":
                    model = MB_LE(str(data_file), t_start=0, t_end=0.5,
                                  learned=False, output_dir=phase_dir)
                else:
                    model = MB_LE(str(data_file), learned=True, w_kc2kc=weights,
                                  output_dir=phase_dir)
                model.run_sim(make_plots=False)
                check_recording(phase_dir, model.sim_t)
                if phase == "learning":
                    weights = generate_pnlist(np.asarray(model.learned_w_kc2kc))
                    assert weights, "学习后没有非零连接，停止测试"
                record["completed_phases"].append(phase)
            finally:
                if initialized:
                    sim.end()  # 学习与测试之间释放硬件；异常时也执行。
        record["status"] = "COMPLETED"
        record["interpretation"] = "Single-file learning/testing only; route recognition is not yet validated."
    except Exception as error:
        record["status"] = "FAILED"
        record["error"] = str(error)
        raise
    finally:
        (output_dir / "run_summary.json").write_text(json.dumps(record, indent=2))
        shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
        print("Outputs: " + str(output_dir) + ".zip", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data/p01.mat")
    parser.add_argument("--run", action="store_true", help="运行原规模学习和测试")
    args = parser.parse_args()
    data_file = args.data.resolve()
    input_info = check_input(data_file)
    if args.run:
        run_case(data_file, Path.cwd() / "outputs", input_info)
    else:
        print("Input check only; no model import or hardware allocation.")


if __name__ == "__main__":
    main()
