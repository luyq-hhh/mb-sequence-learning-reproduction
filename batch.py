"""26 个文件：逐个学习前半段，再测试全部文件；默认只检查输入。"""
import argparse
import json
from pathlib import Path

from run import BASE_DIR, TIME_SCALE_FACTOR, check_input, run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--run", action="store_true", help="执行 26 次学习和 676 次测试")
    parser.add_argument("--learning-file", help="仅学习指定文件，仍测试全部 26 个文件")
    parser.add_argument("--allow-packet-loss", action="store_true",
                        help="记录输入缓冲丢包，但允许继续测试")
    args = parser.parse_args()
    files = sorted(args.data_dir.resolve().glob("*.mat"))
    assert len(files) == 26, "本次批量实验应有 26 个 MAT 文件"
    inputs = [check_input(file) for file in files]
    selected = [file for file in files if args.learning_file in (None, file.name)]
    assert selected, "指定的学习文件不在这 26 个 MAT 文件中"
    record = {"status": "PLANNED", "inputs": inputs, "learning_count": len(selected),
              "testing_count": len(selected) * len(files), "completed_learning_files": [],
              "allow_packet_loss": args.allow_packet_loss}
    model_ms = (sum(item["testing_ms"] for item in inputs) * len(selected)
                + sum(item["learning_ms"] for file, item in zip(files, inputs)
                      if file in selected))
    print("Planned simulations:", len(selected) * (len(files) + 1), flush=True)
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
            if data_file not in selected:
                continue
            print("Learning file:", data_file.name, flush=True)
            # 每个学习文件有独立目录和 ZIP；任一次失败便退出整个批量任务。
            run_case(data_file, output_dir / ("learn_" + data_file.stem), input_info,
                     testing_files=files, allow_packet_loss=args.allow_packet_loss)
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
