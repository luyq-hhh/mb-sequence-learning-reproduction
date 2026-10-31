"""26 个文件：逐个学习前半段，再测试全部文件；默认只检查输入。"""
import argparse
import json
from pathlib import Path

from run import BASE_DIR, TIME_SCALE_FACTOR, check_input, run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--run", action="store_true", help="执行 26 次学习和 676 次测试")
    args = parser.parse_args()
    files = sorted(args.data_dir.resolve().glob("*.mat"))
    assert len(files) == 26, "本次批量实验应有 26 个 MAT 文件"
    inputs = [check_input(file) for file in files]
    record = {"status": "PLANNED", "inputs": inputs, "learning_count": len(files),
              "testing_count": len(files) ** 2, "completed_learning_files": []}
    model_ms = sum(item["testing_ms"] for item in inputs) * (len(files) + 0.5)
    print("Planned simulations:", len(files) * (len(files) + 1), flush=True)
    print("Simulation-only wall hours:", model_ms * TIME_SCALE_FACTOR / 3600000,
          "(deployment and download excluded)", flush=True)
    if not args.run:
        print("Input check only; no model import or hardware allocation.")
        return

    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=False)  # 不覆盖之前的结果。
    record["status"] = "RUNNING"
    try:
        for data_file, input_info in zip(files, inputs):
            print("Learning file:", data_file.name, flush=True)
            # 每个学习文件有独立目录和 ZIP；任一次失败便退出整个批量任务。
            run_case(data_file, output_dir / ("learn_" + data_file.stem), input_info,
                     testing_files=files)
            record["completed_learning_files"].append(data_file.name)
            (output_dir / "batch_summary.json").write_text(json.dumps(record, indent=2))
        record["status"] = "COMPLETED"
    except Exception as error:
        record["status"] = "FAILED"
        record["error"] = str(error)
        raise
    finally:
        (output_dir / "batch_summary.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
