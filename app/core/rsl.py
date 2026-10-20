"""作业新增核心：程序批改 → 最多两次修订 → 验证后保存经验。

模型权重不变；学习对象只是后续提示词中的经验。没有模型路由学习或候选搜索。
"""
import asyncio
import hashlib
import json
import random
import re
import time
import uuid
from pathlib import Path

DATASET_VERSION = "orders-v4-20261020"

def tasks():
    """固定三个互不重叠的数据集。测试题不参与经验筛选。"""
    result = []
    for split, seed, count in [
        ("train", 2026102001, 12), ("validation", 2026102012, 8), ("test", 2026102023, 20)
    ]:
        rng = random.Random(seed)
        for i in range(count):
            rows = [
                dict(id=f"R{k:02}", active=rng.choice([True, True, False]),
                     region=rng.choice(["east", "west"]), qty=rng.randint(1, 12),
                     price=rng.randint(11, 99))
                for k in range(12)
            ]
            region, threshold = rng.choice(["east", "west"]), rng.randint(3, 7)
            prompt = (
                f'只保留 active=true、region="{region}"、qty>={threshold} 的订单。'
                '按qty降序、同qty按id升序排列。最终答案为JSON对象：'
                'ids为排列后的id数组，count为订单数，total为所选订单qty×price之和。'
                '最终对象只允许这三个字段，count和total必须为整数。'
                '先逐行核对筛选条件，列出入选订单及每项qty×price的计算，再求和。'
                '计算过程写在前面，最后用一个```json代码块给出最终对象，代码块后不要附加文字。\n订单：'
                + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            )
            result.append(dict(id=f"{split}-{i+1:02}", split=split, prompt=prompt,
                               rows=rows, region=region, threshold=threshold))
    return result


def expected(task):
    """标准答案由Python直接计算，不请模型给自己打分。"""
    selected = [r for r in task["rows"] if r["active"]
                and r["region"] == task["region"] and r["qty"] >= task["threshold"]]
    selected.sort(key=lambda r: (-r["qty"], r["id"]))
    return dict(ids=[r["id"] for r in selected], count=len(selected),
                total=sum(r["qty"] * r["price"] for r in selected))


def parse_json(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("重复字段")
            obj[key] = value
        return obj
    return json.loads(text, object_pairs_hook=unique)


def normalize_answer(answer):
    """仅拆除明确的末尾JSON代码块；不修改字段、数值，也不执行计算过程。"""
    if not isinstance(answer, str):
        return answer, False
    match = re.fullmatch(r"(?:(?!```).)*```json[ \t]*\r?\n(.*?)\r?\n```[ \t\r\n]*",
                         answer, flags=re.DOTALL)
    return (match.group(1), True) if match else (answer, False)


def evaluate(task, answer):
    normalized, postprocessed = normalize_answer(answer)
    return _evaluate_strict(task, normalized) | {
        "normalized_answer": normalized, "postprocessed": postprocessed,
    }


def _evaluate_strict(task, answer):
    """五项检查，每项0.2分。反馈只指出错误规则，不泄露标准答案数值。"""
    try:
        obj = parse_json(answer)
    except (ValueError, TypeError):
        return dict(score=0.0, passed=False, feedback=[
            "最终答案必须是一个合法JSON对象；计算过程放在唯一的末尾json代码块之前，不要重复字段。"])
    valid = (isinstance(obj, dict) and set(obj) == {"ids", "count", "total"}
             and isinstance(obj["ids"], list) and all(type(x) is str for x in obj["ids"])
             and type(obj["count"]) is int and type(obj["total"]) is int)
    if not valid:
        return dict(score=0.2, passed=False,
                    feedback=["必须且只含ids字符串数组、count整数、total整数，布尔值不是整数。"])
    gold = expected(task)
    checks = [obj[key] == gold[key] for key in ("ids", "count", "total")]
    tips = [
        "ids有误：复核三个过滤条件，按qty降序、同qty按id升序，不能遗漏或重复。",
        "count有误：重新统计符合全部条件的订单数量。",
        "total有误：只对入选订单逐项计算qty乘price，再求和。",
    ]
    return dict(score=(2 + sum(checks)) / 5, passed=all(checks),
                feedback=[tip for ok, tip in zip(checks, tips) if not ok])

def repair_feedback(task, row, round_index):
    """训练时使用带标签的字段反馈；其他集合只给错误类别，不给答案值。"""
    feedback = list(row["evaluation"]["feedback"])
    if task["split"] == "train":
        try:
            actual = parse_json(normalize_answer(row["response"]["answer"])[0])
        except (ValueError, TypeError):
            actual = {}
        actual = actual if isinstance(actual, dict) else {}
        for key, target in expected(task).items():
            if type(actual.get(key)) is not type(target) or actual.get(key) != target:
                feedback.append(f"训练标签 {key}：当前值={json.dumps(actual.get(key))}；"
                                f"正确值={json.dumps(target)}。")
    method = (
        "重新逐行检查active、region、qty三个条件；列齐入选行后再排序、计数和计算金额。"
        if round_index == 1 else
        "上次修正仍未通过。重新从原始数据计算，不沿用之前的合计；"
        "逐项核对数量乘单价，检查边界条件和相同数量时的编号顺序。"
    )
    return ("程序检查发现：\n" + "\n".join(feedback) + "\n" + method
            + "\n先写复核计算过程，最后在唯一的json代码块中输出完整答案。")


def messages_for(task, memory):
    messages = []
    if memory:
        messages.append(dict(role="system", content="以下是验证过的经验；任务原有规则优先：\n"
                             + "\n".join(memory)))
    messages.append(dict(role="user", content=task["prompt"]))
    return messages


async def call_model(generate, messages):
    """所有调用留痕；失败停止，不把接口异常当作模型的低质量答案。"""
    start = time.monotonic()
    try:
        response = await asyncio.wait_for(generate(messages), timeout=120)
        if not isinstance(response.get("answer"), str):
            raise ValueError("缺少文本答案")
        return dict(status="ok", response=response, messages=messages,
                    seconds=round(time.monotonic() - start, 3))
    except Exception as exc:
        # 不保存上游异常全文，避免其中包含凭据或请求头。
        return dict(status="error", error_type=type(exc).__name__, messages=messages,
                    seconds=round(time.monotonic() - start, 3))


async def refine(task, generate, memory=(), max_calls=3):
    if not 1 <= max_calls <= 3:
        raise ValueError("每题只能调用1至3次")
    base = messages_for(task, memory)
    history, best = [], None
    messages = list(base)
    stop = "call_limit"
    for index in range(max_calls):
        if history:
            previous = history[-1]
            messages += [
                dict(role="assistant", content=previous["response"]["answer"]),
                dict(role="user", content=repair_feedback(task, previous, index)),
            ]
        row = await call_model(generate, list(messages))
        row["round"] = index
        history.append(row)
        if row["status"] == "error":
            stop = "upstream_error"
            break
        row["evaluation"] = evaluate(task, row["response"]["answer"])
        row["accepted"] = best is None or row["evaluation"]["score"] > best["evaluation"]["score"]
        if row["accepted"]:
            best = row
        if row["evaluation"]["passed"]:
            stop = "passed"
            break
    return dict(task_id=task["id"], memory=list(memory), history=history,
                feedback_kind="labeled_training" if task["split"] == "train" else "rule_only",
                passed=bool(best and best["evaluation"]["passed"]),
                best_answer=best["response"]["answer"] if best else None,
                best_score=best["evaluation"]["score"] if best else 0.0, stop_reason=stop)


async def validate_rule(generate, memory, rule):
    """同一组验证题：分别使用旧经验、旧经验+新规则；不对验证题执行纠错。"""
    before, after = [], []
    for task in (t for t in tasks() if t["split"] == "validation"):
        before.append(await refine(task, generate, memory, max_calls=1))
        after.append(await refine(task, generate, [*memory, rule], max_calls=1))
    complete = all(r["stop_reason"] != "upstream_error" for r in before + after)
    old = sum(r["passed"] for r in before)
    new = sum(r["passed"] for r in after)
    no_regression = all(not a["passed"] or b["passed"] for a, b in zip(before, after))
    return dict(accepted=complete and no_regression and new > old, complete=complete,
                before_passed=old, after_passed=new, count=len(before),
                no_regression=no_regression, before=before, after=after)


async def learn(task, generate, memory):
    """只从训练题的成功修订中提炼规则。最多5条；不主动修改模型或评价器。"""
    if task["split"] != "train":
        raise ValueError("只有训练题可以学习")
    result = dict(task_id=task["id"], accepted=False, memory_before=list(memory),
                  memory_after=list(memory))
    if len(memory) >= 5:
        return result | {"reason": "memory_full"}
    run = await refine(task, generate, memory)
    result["training"] = run
    first = run["history"][0]
    if not run["passed"] or first.get("evaluation", {}).get("passed", False):
        return result | {"reason": "no_successful_repair"}
    generated = await call_model(generate, [
        dict(role="system", content='从纠错过程总结一条简短通用规则。只返回{"rule":"规则"}。'
             '不含具体订单ID、题号或答案数值，不改变任务规则。'),
        dict(role="user", content=json.dumps(dict(
            task=task["prompt"], wrong=first["response"]["answer"],
            corrected=run["best_answer"], feedback=first["evaluation"]["feedback"]
        ), ensure_ascii=False)),
    ])
    result["proposal"] = generated
    try:
        rule = parse_json(generated["response"]["answer"])["rule"]
    except (KeyError, TypeError, ValueError):
        return result | {"reason": "invalid_proposal"}
    if (not isinstance(rule, str) or not 1 <= len(rule.strip()) <= 240
            or any(c.isdigit() for c in rule) or rule.strip() in memory):
        return result | {"reason": "invalid_or_duplicate_rule"}
    rule = rule.strip()
    trial = await validate_rule(generate, memory, rule)
    result.update(rule=rule, validation=trial, accepted=trial["accepted"],
                  reason="accepted" if trial["accepted"] else "validation_rejected")
    if trial["accepted"]:
        result["memory_after"] = [*memory, rule]
    return result


async def compare(task, generate, memory):
    """基线与反馈组共享同一个首答；经验组单次作答，不利用测试答案学习。"""
    feedback = await refine(task, generate, max_calls=3)
    first = feedback["history"][0]
    baseline = dict(passed=first.get("evaluation", {}).get("passed", False), history=[first])
    if memory:
        remembered = await refine(task, generate, memory, max_calls=1)
    else:
        remembered = dict(baseline, shared_with_baseline=True)
    return dict(task_id=task["id"], memory=list(memory), baseline=baseline,
                feedback=feedback, memory_single=remembered)


class Store:
    """每个调用身份、模型各存一份经验。单进程运行，写入由接口锁串行化。"""
    def __init__(self, root):
        self.root = Path(root)
        self.lock = asyncio.Lock()

    def folder(self, scope, model):
        digest = hashlib.sha256(json.dumps([scope, model], sort_keys=True).encode()).hexdigest()
        return self.root / digest

    def memory(self, scope, model):
        path = self.folder(scope, model) / "memory.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def save(self, scope, model, record, memory=None):
        folder = self.folder(scope, model)
        folder.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        record["run_id"] = run_id
        (folder / f"{run_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        if memory is not None:
            temp = folder / "memory.tmp"
            temp.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(folder / "memory.json")
        return record
