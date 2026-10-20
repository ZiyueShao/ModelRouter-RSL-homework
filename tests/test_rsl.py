import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.rsl import DATASET_VERSION, Store, compare, evaluate, expected, learn, refine, tasks, validate_rule


def run(coro):
    return asyncio.run(coro)


def output(obj):
    return {"answer": json.dumps(obj, ensure_ascii=False)}


def test_verifier_known_answer_and_strict_schema():
    task = dict(region="east", threshold=3, rows=[
        dict(id="B", active=True, region="east", qty=3, price=10),
        dict(id="A", active=True, region="east", qty=3, price=20),
        dict(id="C", active=False, region="east", qty=9, price=99),
        dict(id="D", active=True, region="west", qty=8, price=99),
    ])
    gold = {"ids": ["A", "B"], "count": 2, "total": 90}
    assert expected(task) == gold
    assert evaluate(task, json.dumps(gold))["passed"]
    assert not evaluate(task, '{"ids":[],"count":0,"total":0,"total":90}')["passed"]
    assert not evaluate(task, json.dumps(gold | {"count": True}))["passed"]
    wrapped = evaluate(task, "核算`qty`和`price`。\n```json\n" + json.dumps(gold) + "\n```")
    assert wrapped["passed"] and wrapped["postprocessed"]
    assert json.loads(wrapped["normalized_answer"]) == gold
    assert not evaluate(task, "```json\n" + json.dumps(gold) + "\n```\n额外文字")["passed"]
    assert not evaluate(task, "```json\n{}\n```\n```json\n" + json.dumps(gold) + "\n```")["passed"]
    assert not evaluate(task, json.dumps(gold | {"extra": 1}))["passed"]


def test_data_splits_are_fixed_and_disjoint():
    data = tasks()
    assert data == tasks()
    assert len({t["id"] for t in data}) == 40
    assert len({t["prompt"] for t in data}) == 40
    assert [sum(t["split"] == s for t in data) for s in ("train", "validation", "test")] == [12, 8, 20]


def test_repair_shares_initial_and_feedback_has_no_gold_value():
    task, calls = tasks()[-1], []
    async def fake(messages):
        calls.append(messages)
        return output(expected(task) if len(calls) > 1 else expected(task) | {"total": -1})
    result = run(compare(task, fake, []))
    assert not result["baseline"]["passed"] and result["feedback"]["passed"]
    assert result["baseline"]["history"][0] == result["feedback"]["history"][0]
    assert result["memory_single"]["shared_with_baseline"]
    assert len(calls) == 2
    assert str(expected(task)["total"]) not in calls[1][-1]["content"]


def test_regression_keeps_best_and_budget_stops():
    task, answers = tasks()[0], iter([output(expected(tasks()[0]) | {"total": -1}),
                                      {"answer": "not json"}, {"answer": "{}"}])
    async def fake(messages):
        return next(answers)
    result = run(refine(task, fake))
    assert len(result["history"]) == 3
    assert result["best_score"] == .8
    assert result["stop_reason"] == "call_limit"

def test_labeled_feedback_is_training_only_and_rounds_are_distinct():
    for split in ("train", "validation", "test"):
        task = next(t for t in tasks() if t["split"] == split)
        calls = []
        async def fake(messages):
            calls.append(messages)
            return output(expected(task) | {"total": -1})
        result = run(refine(task, fake))
        assert len(calls) == 3
        assert [len(m) for m in calls] == [1, 3, 5]
        assert calls[1][-1]["content"] != calls[2][-1]["content"]
        feedback = calls[1][-1]["content"]
        assert ("正确值=" in feedback) == (split == "train")
        assert (str(expected(task)["total"]) in feedback) == (split == "train")
        assert len(result["history"][0]["messages"]) == 1


def test_failure_does_not_become_answer_or_leak_exception():
    async def fail(messages):
        raise RuntimeError("private provider detail")
    result = run(refine(tasks()[0], fail))
    assert result["stop_reason"] == "upstream_error"
    assert len(result["history"]) == 1 and not result["passed"]
    assert "private provider detail" not in json.dumps(result)


def task_from_messages(messages):
    return next(t for t in tasks() if any(m["content"] == t["prompt"] for m in messages))


def test_rule_is_saved_only_after_validation_improves():
    async def fake(messages):
        if messages[0]["role"] == "system" and "总结一条" in messages[0]["content"]:
            return output({"rule": "先筛选，再排序，逐项核算金额。"})
        task = task_from_messages(messages)
        fixed = len(messages) > 1
        return output(expected(task) if fixed else expected(task) | {"total": -1})
    result = run(learn(tasks()[0], fake, []))
    assert result["accepted"]
    assert result["validation"]["before_passed"] == 0
    assert result["validation"]["after_passed"] == 8
    assert len(result["memory_after"]) == 1
    assert all(row["task_id"].startswith("validation-")
               for row in result["validation"]["after"])


@pytest.mark.parametrize("case", ["regression", "equal", "failure"])
def test_rule_rejected_on_regression_equal_score_or_failure(case):
    async def fake(messages):
        task = task_from_messages(messages)
        new = messages[0]["role"] == "system"
        index = int(task["id"].split("-")[1])
        if case == "failure" and new:
            raise TimeoutError()
        good = True if case == "equal" else (index != 1 if new else index == 1)
        return output(expected(task) if good else expected(task) | {"total": -1})
    result = run(validate_rule(fake, [], "先核对过滤条件。"))
    assert not result["accepted"]


def test_no_learning_from_test_and_no_rule_from_already_correct():
    async def fake(messages):
        return output(expected(task_from_messages(messages)))
    with pytest.raises(ValueError):
        run(learn(tasks()[-1], fake, []))
    result = run(learn(tasks()[0], fake, []))
    assert not result["accepted"] and result["reason"] == "no_successful_repair"


def test_persistence_scope_and_model_isolation(tmp_path):
    store = Store(tmp_path)
    store.save(["user1"], "m1", {"accepted": True}, memory=["核对条件。"])
    restored = Store(tmp_path)
    assert restored.memory(["user1"], "m1") == ["核对条件。"]
    assert restored.memory(["user2"], "m1") == []
    assert restored.memory(["user1"], "m2") == []


def test_actual_modelrouter_api_auth_policy_dispatch_and_restart(tmp_path, monkeypatch):
    """运行真实ModelRouter启动/鉴权/Provider路径，HTTP上游用MockTransport，绝不出网。"""
    import httpx
    from app.main import app
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    for key, value in {
        "ROUTER_RUNTIME_MODE": "internal", "ROUTER_ENABLE_ADMIN_API": "false",
        "ROUTER_ENABLE_STANDALONE_KEYS": "false", "ROUTER_INTERNAL_API_KEY": "test-local-token",
        "ROUTER_DB": str(tmp_path / "router.db"), "RSL_OUTPUT": str(tmp_path / "rsl"),
        "OPENAI_BASE_URL": "http://mock.invalid/v1", "OPENAI_KEY_1": "mock",
        "HOMEWORK_MODEL": "mock-model", "AUTO_SYNC_PROVIDER_MODELS": "false",
    }.items():
        monkeypatch.setenv(key, value)
    calls = []
    def transport(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "mock-model"
        task = task_from_messages(body["messages"])
        return httpx.Response(200, json={
            "id": "mock", "choices": [{"message": {"role": "assistant",
                                                "content": json.dumps(expected(task))}}],
        })
    original_client = httpx.AsyncClient
    def client_factory(*args, **kwargs):
        return original_client(*args, **kwargs, transport=httpx.MockTransport(transport))
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    headers = {"Authorization": "Bearer test-local-token"}
    with TestClient(app) as client:
        assert client.get("/v1/rsl/tasks").status_code == 401
        assert len(client.get("/v1/rsl/tasks", headers=headers).json()) == 40
        assert client.post("/v1/rsl/run", headers=headers, json={
            "task_id": "test-01", "mode": "learn"}).status_code == 422
        assert client.post("/v1/rsl/run", headers=headers | {"X-Allowed-Models": "other"}, json={
            "task_id": "train-01"}).status_code == 403
        response = client.post("/v1/rsl/run", headers=headers, json={"task_id": "train-01"})
        assert response.status_code == 200 and response.json()["passed"]
        assert len(calls) == 1
        app.state.rsl.save(["internal", "internal", None, None], f"homework:mock-model:{DATASET_VERSION}",
                           {"test": True}, memory=["核对字段。"])
    with TestClient(app) as client:
        assert client.get("/v1/rsl/memory", headers=headers).json()["rules"] == ["核对字段。"]
        assert client.get("/v1/rsl/memory", headers=headers | {"X-User-Id": "other"}).json()["rules"] == []
