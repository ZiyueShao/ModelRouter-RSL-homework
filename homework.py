"""只用这个入口：demo离线说明、serve启动、learn学习、evaluate真实对照。"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from app.core.rsl import DATASET_VERSION, expected, refine, tasks

ROOT = Path(__file__).resolve().parent


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


async def demo():
    """人为先错后对的假模型，只演示程序机制；没有模型服务和网络调用。"""
    task = tasks()[0]
    async def fake(messages):
        answer = expected(task)
        if len(messages) == 1:
            answer["total"] = -1
        return {"answer": json.dumps(answer), "model": "SIMULATED"}
    result = await refine(task, fake)
    result["simulated"] = True
    for row in result["history"]:
        print(f'第{row["round"]}轮：{row["response"]["answer"]}')
        print("检查：", row["evaluation"])
    save(ROOT / "results" / "offline-demo.json", result)
    print("以上是人为设计的假模型演示，不是实际模型效果。")


def request_run(body):
    import httpx
    token = os.environ.get("ROUTER_INTERNAL_API_KEY", "")
    if not token or token == "replace-with-your-local-access-token":
        raise SystemExit("请先在.env中设置本地访问口令，并启动serve。")
    # 此请求固定发往本机，不能被Windows的系统HTTP代理转发。
    with httpx.Client(timeout=1500, trust_env=False) as client:
        response = client.post("http://127.0.0.1:8000/v1/rsl/run", json=body,
                               headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        return response.json()


def has_error(obj):
    if isinstance(obj, dict):
        return obj.get("status") == "error" or any(has_error(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_error(v) for v in obj)
    return False


def experiment(command):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = ROOT / "results" / f"{command}-{stamp}.json"
    if path.exists():
        raise SystemExit("同名结果已存在，请稍后再运行。")
    dataset = [t for t in tasks() if t["split"] == ("train" if command == "learn" else "test")]
    result = dict(simulated=False, command=command, dataset_version=DATASET_VERSION,
                  planned=len(dataset), complete=False, records=[])
    save(path, result)
    frozen_memory = None
    try:
        for task in dataset:
            row = request_run(dict(task_id=task["id"], mode="learn" if command == "learn" else "compare"))
            result["records"].append(row)
            save(path, result)
            print(f'{task["id"]} 已记录；结果保存在 {path.name}', flush=True)
            if has_error(row):
                result["stop_reason"] = "upstream_error"
                break
            if command == "evaluate":
                if frozen_memory is None:
                    frozen_memory = row["memory"]
                elif frozen_memory != row["memory"]:
                    result["stop_reason"] = "memory_changed_during_test"
                    break
        else:
            result["complete"] = True
    except Exception as exc:
        result["stop_reason"] = type(exc).__name__
        print("调用中断，已保存完成的部分；检查服务与模型配置。")
    finally:
        if command == "evaluate" and result["records"]:
            result["summary"] = {
                group: {
                    "passed": sum(r[group]["passed"] for r in result["records"]),
                    "count": len(result["records"]),
                    "mean_calls": sum(len(r[group]["history"]) for r in result["records"]) / len(result["records"]),
                    "mean_seconds": sum(sum(h["seconds"] for h in r[group]["history"])
                                        for r in result["records"]) / len(result["records"]),
                }
                for group in ("baseline", "feedback", "memory_single")
            }
        save(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))
    print(f"完整记录：{path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "serve", "learn", "evaluate"])
    args = parser.parse_args()
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ["NO_PROXY"] = ",".join(filter(None, [
        os.environ.get("NO_PROXY", ""), "127.0.0.1", "localhost"
    ]))
    if args.command == "demo":
        asyncio.run(demo())
    elif args.command == "serve":
        token = os.environ.get("ROUTER_INTERNAL_API_KEY", "")
        if not token or token == "replace-with-your-local-access-token":
            raise SystemExit("先复制.env.example为.env，并设置自己的本地访问口令。")
        if not all(os.environ.get(k) for k in ("OPENAI_BASE_URL", "OPENAI_KEY_1", "HOMEWORK_MODEL")):
            raise SystemExit("请填写OPENAI_BASE_URL、OPENAI_KEY_1、HOMEWORK_MODEL。")
        Path(os.environ.get("ROUTER_DB", "results/router.db")).parent.mkdir(parents=True, exist_ok=True)
        import uvicorn
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, workers=1)
    else:
        print("即将调用真实模型；付费API会产生费用。中断可按Ctrl+C。", flush=True)
        experiment(args.command)


if __name__ == "__main__":
    main()
