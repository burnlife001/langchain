# /// script
# requires-python = ">=3.10"
# dependencies = ["langchain-openai"]
# ///
# -*- coding: utf-8 -*-
"""四轮 LLM 审校流水线：同音字错别字 + 段落排版检查。

用法(在脚本所在目录执行):
    uv run --script review_pipeline.py <input_file> [out_dir] [prompt_file] [rounds]

示例:
    uv run --script review_pipeline.py test.txt test_out.txt 2

流程:
    输入文件 -> 第1轮 review -> file_out1.md -> 第2轮 review -> file_out2.md -> ... -> file_outN.md
    每一轮都是全新上下文（不携带历史对话），输入为上一轮的输出全文。

参数:
    input_file    待审校的文件
    out_dir       输出目录，默认为输入文件所在目录下的 test_out 目录
    prompt_file   可选，自定义系统提示词文件（UTF-8 文本），不传则使用内置校对提示词
    rounds        审校轮次，默认为 2，必须 >= 1
    （prompt_file 与 rounds 顺序不限：整数识别为 rounds，其余按文件路径解析）

环境变量:
    BASE_URL              OpenAI 兼容端点
    LLM_MODEL             模型名
    SILICONFLOW_API_KEY   API key（或 OPENAI_API_KEY）
    REVIEW_MAX_TOKENS     可选，单轮最大输出 token，默认 8192
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROUNDS = 2

SYSTEM_PROMPT = """你是一名中文文稿校对编辑，只做两类检查，不要改写内容风格、不要增删信息：

1. 同音字错别字：修正因拼音相同/相近而误用的汉字（如「的/得/地」「做/作」「在/再」「已/以」等），结合上下文判断。
2. 段落排版：修复明显的排版问题——多余空行/缺少空行、段落首行缩进不一致、全半角标点混用（统一为中文全角标点）、行尾多余空格。
输出要求：直接输出修正后的完整全文，保持原有的 Markdown 结构与标题层级，不要输出解释、修改清单或 diff。除上述两类问题外，一字不改。"""


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


def review_round(model, text: str, round_no: int, system_prompt: str) -> str:
    """单轮审校：全新上下文，输入全文，返回修正后的全文。"""
    print(f"[round {round_no}] input {len(text)} chars, calling {model.model} ...", flush=True)
    response = model.invoke(
        [
            ("system", system_prompt),
            ("human", f"请审校以下全文，输出修正后的完整全文：\n\n{text}"),
        ]
    )
    result = response.text if hasattr(response, "text") else str(response.content)
    print(f"[round {round_no}] output {len(result)} chars", flush=True)
    return result


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__ or "usage: uv run --script review_pipeline.py <input_file> [out_dir] [prompt_file] [rounds]")

    src = Path(sys.argv[1])
    if not src.is_file():
        sys.exit(f"input file not found: {src}")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / "test_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = ROUNDS
    system_prompt = SYSTEM_PROMPT
    for arg in sys.argv[3:5]:
        try:
            parsed = int(arg)
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed < 1:
                sys.exit(f"rounds must be >= 1, got: {parsed}")
            rounds = parsed
            continue
        prompt_src = Path(arg)
        if not prompt_src.is_file():
            sys.exit(f"prompt file not found: {prompt_src}")
        system_prompt = prompt_src.read_text(encoding="utf-8")
        print(f"using custom prompt: {prompt_src} ({len(system_prompt)} chars)", flush=True)

    model = build_model()
    current = src.read_text(encoding="utf-8")

    for i in range(1, rounds + 1):
        current = review_round(model, current, i, system_prompt)
        out_file = out_dir / f"file_out{i}.md"
        out_file.write_text(current, encoding="utf-8")
        print(f"[round {i}] written: {out_file}", flush=True)

    print(f"done: {rounds} rounds completed")


if __name__ == "__main__":
    main()
