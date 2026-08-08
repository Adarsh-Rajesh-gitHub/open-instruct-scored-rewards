import random

from projects.student_state_tutor import misconception_probe


def test_problem_options_are_distinct_and_cover_labels():
    rng = random.Random(0)
    for index in range(100):
        problem = misconception_probe.make_problem(rng, index)
        assert {option["label"] for option in problem["options"]} == set(misconception_probe.LABELS)
        assert len({option["value"] for option in problem["options"]}) == len(misconception_probe.LABELS)


def test_parse_choice_uses_declared_answer():
    problem = misconception_probe.make_problem(random.Random(1), 0)
    assert misconception_probe.parse_choice("I added the values.\nAnswer: C", problem) == "C"
    assert misconception_probe.parse_choice("The correct answer is A. 5.", problem) == "A"


def test_parse_choice_can_resolve_numeric_answer():
    problem = misconception_probe.make_problem(random.Random(2), 1)
    selected = problem["options"][0]
    response = f"My answer is {selected['value']}"
    assert misconception_probe.parse_choice(response, problem) == selected["letter"]


def test_unparseable_response_is_unclassified():
    problem = misconception_probe.make_problem(random.Random(3), 2)
    assert misconception_probe.parse_choice("I do not know.", problem) is None
