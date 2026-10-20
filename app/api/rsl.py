"""接入层：沿用ModelRouter的身份验证、模型权限和Provider，不另建路由学习器。"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.auth import resolve_request_context
from app.core.request_context import RequestContext
from app.core.router import DISABLED_ROUTING_KEYS
from app.core.routing_policy import load_routing_policy
from app.core.rsl import DATASET_VERSION, compare, learn, refine, tasks
from app.schemas.chat import ChatMessage, ChatRequest

router = APIRouter(prefix="/v1/rsl", tags=["作业A：反馈与经验"])


class RunRequest(BaseModel):
    task_id: str
    model: str = Field(default="homework", min_length=1, max_length=120)
    mode: Literal["single", "feedback", "learn", "compare"] = "feedback"
    use_memory: bool = True


def scope_for(context):
    return [context.subject_type, context.subject_id, context.user_id, context.routing_profile_id]


async def model_for(request, context, key):
    engine = request.app.state.router
    policy = await load_routing_policy(request.app.state.db, context.routing_profile_id)
    allowed = policy.effective_allowed_models(context.allowed_models, set(engine.model_map))
    if key not in allowed or key in DISABLED_ROUTING_KEYS or key not in engine.model_map:
        raise HTTPException(403, "模型不存在或没有使用权限")
    return await engine.route(key)


@router.get("/tasks")
async def catalog(context: RequestContext = Depends(resolve_request_context)):
    return [{"id": t["id"], "split": t["split"], "prompt": t["prompt"]} for t in tasks()]


@router.get("/memory")
async def memory(request: Request, model: str = "homework",
                 context: RequestContext = Depends(resolve_request_context)):
    _, config = await model_for(request, context, model)
    # 实际模型名参与隔离，换模型不会意外沿用旧经验。
    identity = f"{model}:{config.name}:{DATASET_VERSION}"
    return {"model": model, "rules": request.app.state.rsl.memory(scope_for(context), identity)}


@router.post("/run")
async def run(body: RunRequest, request: Request,
              context: RequestContext = Depends(resolve_request_context)):
    task = next((t for t in tasks() if t["id"] == body.task_id), None)
    if task is None:
        raise HTTPException(404, "未知题目")
    if body.mode == "learn" and task["split"] != "train":
        raise HTTPException(422, "只能从train题目学习，验证题和测试题禁止更新经验")
    provider, config = await model_for(request, context, body.model)
    store, scope = request.app.state.rsl, scope_for(context)
    identity = f"{body.model}:{config.name}:{DATASET_VERSION}"
    async def generate(messages):
        response = await provider.chat_completions(ChatRequest(
            model=body.model, messages=[ChatMessage(**m) for m in messages],
            max_tokens=2048, temperature=0), config)
        return dict(answer=response.choices[0].message.content, model=response.model,
                    usage=response.usage.model_dump() if response.usage else None)

    async with store.lock:
        saved = store.memory(scope, identity)
        rules = saved if body.use_memory or body.mode == "learn" else []
        if body.mode == "learn":
            result = await learn(task, generate, saved)
        elif body.mode == "compare":
            result = await compare(task, generate, rules)
        else:
            result = await refine(task, generate, rules,
                                  max_calls=1 if body.mode == "single" else 3)
        result.update(mode=body.mode, model=body.model, actual_model=config.name,
                      dataset_version=DATASET_VERSION, simulated=False)
        return store.save(scope, identity, result,
                          memory=result["memory_after"] if body.mode == "learn" else None)
