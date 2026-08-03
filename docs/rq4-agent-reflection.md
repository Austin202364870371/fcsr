# RQ4：如何在基础 Agent 框架中引入 Reflection 机制？

**状态：研究设计，尚未实现。**

## 问题与假设

任务失败后，Agent 需要区分重试当前动作、修改剩余计划和候选 Skill 不足。RQ4 研究证据约束的 Reflection 能否减少重复失败并改善后续 verifier 成功率，而非在每次调用后生成无约束长文本。

假设是仅在明确失败信号下触发反思，并将其限制为结构化、可验证的修复建议，可获得比自由文本反思更低成本、更可审计的收益。

## 触发与输出约束

只在非零工具退出、参数/schema 失败、前置条件不满足、verifier 失败、重复动作或预算风险时触发。成功观察不触发。

```json
{
  "failure_class": "wrong_skill | wrong_parameter | plan_dependency | environment | verifier | budget",
  "evidence_event_ids": ["..."],
  "repair_scope": "retry_action | revise_remaining_plan | stop",
  "proposed_change": "...",
  "confidence": 0.0
}
```

执行器只接受可解析、引用真实事件且不超预算的建议；私有 GT、oracle 和 verifier 内部映射不进入反思上下文。

## 对照与判据

比较无反思、自由文本反思、证据约束反思。三组共享 RQ3 的规划器、任务、Skill 候选、组织、重试上限和工具。反思不能新增候选 Skill；判断缺 Skill 时只能安全终止并记录。

主指标仍是 verifier 成功率，辅助指标包括失败后修复率、重复失败率、反思次数/token、错误分类一致性与修复延迟。人工抽查 trace 仅用于诊断，不能泄漏给在线 Agent。

## 实施顺序

先完成 RQ3 的 trace 与 verifier 事件接口；定义 JSON schema、预算和可见性；实现无反思和自由文本基线，再实现证据约束反思；用固定失败任务集调试后做全任务配对实验。

## 与其他 RQ 的关系

RQ4 是 RQ3 执行闭环的失败处理模块，RQ5 负责其成本、修复率和成功率的统一报告。
