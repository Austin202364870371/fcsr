from agent.llm import LLMReply, PlanStep, SkillPlan


def test_shared_llm_contract_validates_a_plan():
    plan = SkillPlan(
        selected_skill_aliases=("S01",),
        steps=(
            PlanStep(
                id="step-1",
                objective="do",
                skill_aliases=("S01",),
                expected_output="x",
            ),
        ),
        final_output="x",
    )

    assert plan.selected_skill_aliases == ("S01",)
    assert LLMReply(content="{}", prompt_tokens=0, completion_tokens=0).content == "{}"
