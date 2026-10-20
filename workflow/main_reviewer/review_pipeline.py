# /// script
# requires-python = ">=3.10"
# dependencies = ["langchain-openai"]
# ///
# -*- coding: utf-8 -*-
"""主编-审稿双角色修订流水线。

用法(在脚本所在目录执行):
    uv run --script review_pipeline.py [requirement_file] [out_dir] [max_n] [editor_prompt_file] [reviewer_prompt_file]

示例:
    uv run --script review_pipeline.py 需求.txt review_out 3

流程:
    阶段1: 与主编(A)对话生成初稿.md（输入 /done 结束对话，主编最后一条回复存为初稿）
    阶段2: 循环 n = 1..max_n
        审稿人(B)审阅当前稿件 -> 建议(1.n).md
        主编(A)根据建议修订   -> 初稿(1.n+1).md
        主编判定可定稿 -> 保存 定稿.md，流程结束
    达到 max_n 仍未定稿 -> 流程结束，保留最新稿件

参数:
    requirement_file      可选，需求/素材文件，作为对话的第一条用户消息
    out_dir               输出目录，默认为 review_out
    max_n                 最大审稿轮次，默认为 3，必须 >= 1
    editor_prompt_file    可选，自定义主编系统提示词（UTF-8 文本）
    reviewer_prompt_file  可选，自定义审稿人系统提示词（UTF-8 文本）

环境变量:
    BASE_URL              OpenAI 兼容端点
    LLM_MODEL             模型名
    SILICONFLOW_API_KEY   API key（或 OPENAI_API_KEY）
    REVIEW_MAX_TOKENS     可选，单轮最大输出 token，默认 8192
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

MAX_N = 3

EDITOR_PROMPT = """你是一名主编(A)，负责撰写、审阅与修改稿件。

工作流程：
1. 先与用户对话，理解需求后撰写初稿。
2. 收到审稿建议后，逐条判断并吸收合理建议，输出完整修订稿。

要求：
- 初稿与修订稿均输出完整全文，保持 Markdown 结构与标题层级。
- 修订时只吸收审稿建议中合理、与需求一致的部分，不盲从；对不采纳的建议无需解释。
- 除建议要求修改的内容外，保持原文一字不改。
- 每次输出修订稿后，在全文最后一行单独输出定稿判定：`【定稿判定】是` 或 `【定稿判定】否`。"""

REVIEWER_PROMPT = """你是一名审稿人(B)，负责对稿件进行审阅并提出修改建议，不直接改稿。

要求：
- 从内容准确性、结构逻辑、表达清晰度、排版规范四个方面审阅。
- 输出结构化建议清单，每条包含：位置、问题描述、修改建议。
- 按重要程度排序，最重要的在前。
- 没有问题的地方不要强行凑建议。"""

FINAL_MARK = re.compile(r"【定稿判定】\s*(是|否)")


def build_model():
    from langchain_openai import ChatOpenAI

    base_url = os.environ["BASE_URL"]
    model_name = os.environ["LLM_MODEL"]
    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("missing API key: set SILICONFLOW_API_KEY or OPENAI_API_KEY")

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=int(os.environ.get("REVIEW_MAX_TOKENS", "8192")),
        temperature=0.0,
    )


def invoke_text(model, messages: list) -> str:
    """调用模型，返回纯文本回复。"""
    response = model.invoke(messages)
    return response.text if hasattr(response, "text") else str(response.content)


def dialogue_phase(model, requirement: str | None, editor_prompt: str) -> str:
    """阶段1：与主编多轮对话，输入 /done 结束，主编最后一条回复作为初稿。"""
    messages: list = [("system", editor_prompt)]
    if requirement:
        messages.append(("human", requirement))
        print("[对话] 已将需求文件作为首条消息发送给主编", flush=True)

    print("[对话] 开始与主编对话，输入 /done 结束对话并生成初稿", flush=True)
    while True:
        try:
            user_input = input("你> ").strip()
        except EOFError:
            user_input = "/done"
        if user_input.lower() in ("/done", "done", "定稿"):
            break
        if not user_input:
            continue
        messages.append(("human", user_input))
        reply = invoke_text(model, messages)
        messages.append(("assistant", reply))
        print(f"\n主编> {reply}\n", flush=True)

    draft = next(
        (m[1] for m in reversed(messages) if m[0] == "assistant"),
        None,
    )
    if not draft and requirement:
        # 非交互场景：需求文件已给出但对话为空，直接让主编产出初稿
        print("[对话] 无对话记录，由主编根据需求直接生成初稿 ...", flush=True)
        draft = invoke_text(model, messages)
    if not draft:
        sys.exit("对话未产生任何主编回复，无法生成初稿")
    # 对话阶段不保留定稿判定标记
    return FINAL_MARK.sub("", draft).strip()


def review_round(model, draft: str, round_no: int, reviewer_prompt: str) -> str:
    """审稿：B 审阅当前稿件，输出建议。"""
    print(f"[建议 1.{round_no}] input {len(draft)} chars, calling {model.model} ...", flush=True)
    result = invoke_text(
        model,
        [
            ("system", reviewer_prompt),
            ("human", f"请审阅以下稿件，输出修改建议清单：\n\n{draft}"),
        ],
    )
    print(f"[建议 1.{round_no}] output {len(result)} chars", flush=True)
    return result


def revise_round(
    model, draft: str, suggestions: str, round_no: int, editor_prompt: str
) -> tuple[str, bool]:
    """修订：A 根据建议修改稿件，返回 (新稿件, 是否定稿)。"""
    print(f"[初稿 1.{round_no + 1}] revising based on 建议 1.{round_no} ...", flush=True)
    result = invoke_text(
        model,
        [
            ("system", editor_prompt),
            (
                "human",
                "以下是当前稿件与审稿建议，请输出完整修订稿，"
                "并在最后一行输出定稿判定。\n\n"
                f"=== 当前稿件 ===\n{draft}\n\n=== 审稿建议 ===\n{suggestions}",
            ),
        ],
    )
    match = FINAL_MARK.search(result[-200:])
    is_final = bool(match and match.group(1) == "是")
    new_draft = FINAL_MARK.sub("", result).strip()
    print(
        f"[初稿 1.{round_no + 1}] output {len(new_draft)} chars, "
        f"定稿判定: {'是' if is_final else '否'}",
        flush=True,
    )
    return new_draft, is_final


def main() -> None:
    args = sys.argv[1:]

    requirement: str | None = None
    out_dir = Path("review_out")
    max_n = MAX_N
    editor_prompt = EDITOR_PROMPT
    reviewer_prompt = REVIEWER_PROMPT

    int_args = []
    file_args = []
    for arg in args:
        try:
            int_args.append(int(arg))
        except ValueError:
            file_args.append(arg)

    if int_args:
        if int_args[0] < 1:
            sys.exit(f"max_n must be >= 1, got: {int_args[0]}")
        max_n = int_args[0]

    if file_args and Path(file_args[0]).is_file():
        requirement = Path(file_args[0]).read_text(encoding="utf-8")
        file_args = file_args[1:]
    if file_args:
        out_dir = Path(file_args[0])
        file_args = file_args[1:]
    # 依次解析主编/审稿人提示词文件
    remaining = file_args[:2]
    if remaining:
        editor_prompt = Path(remaining[0]).read_text(encoding="utf-8")
        print(f"using custom editor prompt: {remaining[0]} ({len(editor_prompt)} chars)", flush=True)
    if len(remaining) > 1:
        reviewer_prompt = Path(remaining[1]).read_text(encoding="utf-8")
        print(f"using custom reviewer prompt: {remaining[1]} ({len(reviewer_prompt)} chars)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()

    # 阶段1：对话生成初稿
    draft = dialogue_phase(model, requirement, editor_prompt)
    draft_file = out_dir / "初稿.md"
    draft_file.write_text(draft, encoding="utf-8")
    print(f"[初稿] written: {draft_file}", flush=True)

    # 阶段2：审稿-修订循环
    for n in range(1, max_n + 1):
        suggestions = review_round(model, draft, n, reviewer_prompt)
        sug_file = out_dir / f"建议(1.{n}).md"
        sug_file.write_text(suggestions, encoding="utf-8")
        print(f"[建议 1.{n}] written: {sug_file}", flush=True)

        draft, is_final = revise_round(model, draft, suggestions, n, editor_prompt)
        draft_file = out_dir / f"初稿(1.{n + 1}).md"
        draft_file.write_text(draft, encoding="utf-8")
        print(f"[初稿 1.{n + 1}] written: {draft_file}", flush=True)

        if is_final:
            final_file = out_dir / "定稿.md"
            final_file.write_text(draft, encoding="utf-8")
            print(f"[定稿] 主编判定可定稿，written: {final_file}", flush=True)
            return

    print(f"done: 达到 max_n={max_n}，未判定定稿，最新稿件: {draft_file}", flush=True)


if __name__ == "__main__":
    main()
