# langchain / workflow

LLM 驱动的写作与校对流水线。

## 模块

| 目录 | 一句话说明 |
| --- | --- |
| `editor_reviewer/` | 主编(A) + 审稿人(B) 双角色交互式流水：先多轮对话产出初稿，再循环“提建议 → 修订”直到主编判定定稿，最多 `max_n` 轮。 |
| `reviewer/` | 单角色无状态多轮流水：每轮以全新上下文对全文做同音字错别字 + 段落排版校对，连续输出 `file_outN.md`，跑 `rounds` 轮。 |

## 公共环境变量

两个流水线都通过 `langchain-openai` 调用 OpenAI 兼容端点，需要：

- `BASE_URL` — OpenAI 兼容端点
- `LLM_MODEL` — 模型名
- `SILICONFLOW_API_KEY` 或 `OPENAI_API_KEY` — API key
- `REVIEW_MAX_TOKENS` — 可选，单轮最大输出 token，默认 `8192`

## 运行

```bash
# 进入子目录执行（依赖由脚本头部 # /// script 声明，uv 自动解析）
cd editor_reviewer
uv run --script review_pipeline.py 需求.txt review_out 3

cd ../reviewer
uv run --script review_pipeline.py test.txt
```

每个脚本都支持：`uv run --script review_pipeline.py [files...] [max_n_or_rounds]`，
文件参数与整数参数不限定顺序，整数自动识别为轮次上限。
