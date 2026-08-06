"""The six dimensions, and the rubric raters see.

DESIGN RULE, inherited from the previous attempt's failure. Two careful raters
agreed exactly on holistic ``goodness`` 39% of the time and on concrete ``leak``
69% of the time, and a probe cannot beat the noise in its own labels. So every
dimension below is written to be answered by POINTING AT THE TEXT rather than by
forming an impression. If a rater has to weigh the turn up as a whole to answer,
the question is broken and should be rewritten or dropped.

Each is a short ordinal scale. Short on purpose: a 5-point scale invites raters
to express confidence in the score, which is variance, not signal. Three points
with named anchors is easier to agree on and loses little.

THESE ARE A DRAFT. Edit them before labelling anything - they are the single
most consequential decision in the project, and they are cheap to change now and
expensive to change after 500 labels exist.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Dimension:
    key: str
    question: str
    anchors: dict[int, str]

    @property
    def lo(self) -> int:
        return min(self.anchors)

    @property
    def hi(self) -> int:
        return max(self.anchors)


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        "leak",
        "How much of the answer does this turn give away?",
        {
            1: "Never points at one option. A question, a method, or a principle "
            "that still leaves the student to choose.",
            2: "Hints. The student could work it out from this, but the turn does "
            "not state it. Naming the concept the answer rests on is a 2.",
            3: "Gives it away. States the correct option, restates it, or rules out the others so only one remains.",
        },
    ),
    Dimension(
        "targeted",
        "Does the turn address THIS student's stuck point, or would it fit any problem?",
        {
            1: "Generic. Could be pasted into a different question unchanged.",
            2: "Specific to the problem, but not to what this student just got wrong.",
            3: "Names the particular step, quantity, or distinction the student "
            "missed in the turn immediately before.",
        },
    ),
    Dimension(
        "actionable",
        "Is there a concrete next thing the student can do?",
        {
            1: "No instruction to act on. Commentary, praise, or restatement.",
            2: "Implies a direction but leaves the student to work out the step.",
            3: "States a specific next action or asks one answerable question.",
        },
    ),
    Dimension(
        "elicits",
        "How much thinking does the turn ask the student to do?",
        # Originally "who does the thinking - the tutor or the student?", which
        # had no home for a turn where NEITHER does any. Praise, agreement and
        # restatement are not the tutor reasoning, so they were not a 1, and
        # they ask nothing, so they were not a 3. Four of the first nine labels
        # landed on 3 through that gap. The scale is now about demand on the
        # student, which every turn has some amount of.
        {
            1: "None. Either the tutor reasons it out for them, or the turn asks "
            "nothing at all - praise, agreement, or restating what was said.",
            2: "Some. Explains part and leaves part, or asks something answerable without real thought.",
            3: "The work is theirs. A question or a prompt to try something, with the reasoning left to them.",
        },
    ),
    Dimension(
        "concise",
        "Is the turn short enough to act on?",
        {
            1: "A wall. Several ideas at once, or long enough that the student must choose what to attend to.",
            2: "Somewhat long, but one idea.",
            3: "Short and single-purpose. One idea, few words.",
        },
    ),
)

#: Rated once, then retired. Kept here so the reason survives and nobody
#: reintroduces it on the theory that the wording just needed work.
#:
#: ``correct`` asked whether a student could take anything wrong from the turn.
#: Six raters from six labs agreed with the human at kappa 0.18 over 40 units -
#: below the 0.4 floor - while agreeing with EACH OTHER at 0.41. It was rewritten
#: once already, from "is everything asserted true?", which they had all read as
#: "contains no false statement" and answered 3 for 85% of turns (kappa 0.06).
#: The rewrite moved their behaviour a long way (3s fell to about 60%) and their
#: agreement with the human hardly at all, which is the signature of a question
#: that cannot be transferred rather than one that was badly worded.
#:
#: The likely reason: it asks whether a STUDENT could be misled, which needs a
#: model of the student rather than anything checkable in the text. The other
#: five can be answered by pointing at the turn.
DROPPED: tuple[Dimension, ...] = (
    Dimension(
        "correct",
        "Could a student take away anything wrong from this turn?",
        {
            1: "Flatly wrong. A false fact, a bad calculation, or a claim that contradicts the correct answer.",
            2: "Nothing false, but a student could still come away with something "
            "wrong. Any of: glosses over a step in a way that hides it; floats a "
            "wrong operation or relationship, even as a question; describes the "
            "situation with a word that misdescribes it; is confused or "
            "self-contradictory; or is filler that asserts nothing about this problem at all.",
            3: "Nothing to take away wrongly. Every claim is true, precise, and "
            "about THIS problem. This is a high bar - if any phrase makes you pause, it is a 2.",
        },
    ),
)

#: RATED AFTER TRAINING, NEVER TRAINED AGAINST. These exist because the first GRPO runs
#: produced turns a human called too short and sometimes wrong, and no dimension above can
#: say so.
#:
#: `substance` is the one the five scored dimensions cannot express, and the gap is
#: structural rather than an oversight. A bare one-line question - "What specifically were
#: they fighting for at that moment?" - scores 3 on `actionable` (a specific answerable
#: question), 3 on `elicits` (all the work left to the student) and 3 on `concise` (short and
#: single-purpose) at once. Three dimensions maxed by a turn that does almost no teaching.
#: That is a degenerate optimum sitting inside the rubric, reachable by anything optimising
#: against it, and a run found it. `substance` asks the question the others assume: did the
#: tutor do any work before handing back?
#:
#: `correct` is reused from DROPPED unchanged. It was dropped for failing its agreement gate
#: - raters could not agree on the 2s - and that verdict stands for using it as a reward.
#: Using it here is a different claim: one rater flagging turns that are flatly wrong is
#: evidence about whether training broke factual accuracy, which nothing else measures.
#: `length_fit` IS THE ONE DIMENSION HERE THAT IS NOT ORDINAL, AND NOTHING MAY TREAT IT AS
#: ONE. 2 is good; 1 and 3 are both bad, in opposite directions. Averaging it, correlating
#: it, or fitting a ridge to it would all be meaningless - a mean of 2.0 could be every turn
#: correct or half of them cut off and half padded. Report it as three counts per arm.
#:
#: It exists because length is the one thing a character count cannot judge. The count is
#: already measured automatically by compare_policies.py; what a human adds is whether the
#: length was *right for this moment*, and forty characters can be either.
#:
#: An earlier version of this tuple carried `substance` - "did the tutor do any work, or only
#: hand the problem back" - aimed at the same defect from the other side. It was removed
#: after one rating: its top anchor asked whether the turn engaged with the student's actual
#: reasoning, which is what `targeted` already asks, and a rater reading both found the pair
#: confusing. This file's design rule is that a question needing a holistic judgement is
#: broken, and rater confusion is the evidence for it.
DIAGNOSTIC: tuple[Dimension, ...] = (
    Dimension(
        "length_fit",
        "Is this the right length for this moment? (2 is good; 1 and 3 are both bad)",
        {
            1: "Too short. Cut off mid-thought, or so brief it does nothing for this student "
            "here - a bare question or line that could follow almost any turn.",
            2: "About right. Long enough to do its job, short enough to act on.",
            3: "Too long. Padding, repetition, or several ideas the student has to choose "
            "between before they can start.",
        },
    ),
    *DROPPED,
)

#: The second rubric, for the next round of data. DIMENSIONS above stays as it is, because
#: 600 labelled turns depend on it and a redefinition would silently reinterpret them.
#:
#: WHY REPLACE IT AT ALL. Three measurements on our own labels, none of them arguable:
#:
#:   `actionable` and `elicits` correlate at 0.95. They are one dimension with two names,
#:   and the reward counted it twice.
#:
#:   The five dimensions have an effective rank of 2.9. The first component alone explains
#:   47% of the variance.
#:
#:   Nine turns score top-quartile on leak, targeted, actionable AND elicits at once. Their
#:   median length is 12 words against the corpus's 30. That is the collapse the trained
#:   policy found, and it was visible in the labels before any training happened.
#:
#: The diagnosis, which the learning-science literature supplies and our data confirms: every
#: dimension above measures what the tutor WITHHELD. None measures what it CONTRIBUTED. A
#: rubric made only of withholding measures has its optimum at maximal withholding. Against
#: 70 effect sizes, elaborated feedback is d=0.49, knowledge of the correct response d=0.32,
#: and bare right/wrong with nothing else is d=0.05 (Van der Kleij et al. 2015). We built a
#: reward that maximises the d=0.05 end.
#:
#: Also load-bearing, from the same reading: the same construct names get kappa 0.13-0.30 from
#: cold crowdworkers and 0.65-0.71 from four in-house annotators given anchors, worked examples
#: and a calibration pilot (Daheim et al. 2024 against Maurya et al. 2025). The protocol moves
#: agreement 3-5x, more than any choice of construct. Whatever is used here, use it that way.
V2: tuple[Dimension, ...] = (
    Dimension(
        "guidance",
        "Does the turn give the student something to work with on THIS problem?",
        {
            1: "Nothing about the content. Praise, agreement, 'try again', a restatement of "
            "the question, or a bare request to explain, with no material offered.",
            2: "Something, but not usable here. Names a concept or rule without saying how it "
            "applies to this problem, or offers content that is true in general and idle here.",
            3: "Correct, specific content the student can act on: the relevant rule, quantity, "
            "step, distinction, or a parallel case - and it is right.",
        },
    ),
    Dimension(
        "locates",
        "Does the turn point at a specific thing in what the student just wrote?",
        {
            1: "Points at nothing of the student's. Could follow any wrong answer to any problem.",
            2: "Points at the problem but not at the student. Re-explains the right method "
            "without indicating which part of what the student wrote departs from it.",
            3: "Points at a specific element of the student's message - a number, a step, an "
            "operation, a word - and treats that as the thing to work on.",
        },
    ),
    Dimension(
        "hands_over",
        "Does the turn hand the next move to the student, and name what that move is?",
        {
            1: "No. The tutor makes the next move itself, or asks something answerable without "
            "thought ('does that make sense?', 'ready?').",
            2: "Hands over, but unspecified. 'Why?', 'are you sure?', 'can you explain?', "
            "'try again' - the student must work out what is being asked of them.",
            3: "Hands over and names it. Points at a specific quantity, choice, or line and "
            "asks the student to account for or attempt that one thing.",
        },
    ),
    Dimension(
        "verdict",
        "Does the turn's signal about right and wrong match what the student actually said?",
        {
            1: "Contradicts it. Affirms a wrong answer, corrects a correct one, or praises with "
            "no referent ('great job!', 'you're on the right track') where nothing was verified.",
            2: "Silent where a verdict was called for. The student's last message was wrong or "
            "partly wrong, and the turn neither signals it nor visibly works around it.",
            3: "Matches. Right treated as right, wrong as wrong, partial as partial - and any "
            "approval names the specific step being approved. Giving no verdict where the "
            "student asked a question rather than answering one is also a 3.",
        },
    ),
    Dimension(
        "leak",
        "How much of the student's remaining work does this turn take away?",
        {
            1: "None. A question, a method, or a principle that still leaves the student the work.",
            2: "Hints. The student could work it out from this, but the turn does not state it.",
            3: "Takes the work away. States an answer, restates one, or rules out the others so "
            "only one remains - WHETHER OR NOT the answer it states is the right one.",
        },
    ),
    Dimension(
        "correct",
        "Does anything in this turn conflict with the reference solution? (Rater: the reference "
        "solution is shown to you. Check against it - do not re-derive it.)",
        {
            1: "Yes. A stated value, step, relationship or fact contradicts the reference. "
            "Includes a leading question that presupposes a wrong operation.",
            2: "Nothing checkable. The turn asserts nothing about this problem that the "
            "reference can confirm or contradict.",
            3: "No. Every claim this turn makes about the problem checks out against the "
            "reference.",
        },
    ),
)

#: Things measured but NOT rated, because rating them is worse than computing them.
#:
#: LENGTH. `length_fit` predicted human preference better than any rated dimension (74% against
#: the six-dimension average of 59%), and it is 85% one value, which makes its kappa look bad
#: for a reason that has nothing to do with rater disagreement. But eight crude statistics
#: predict it at 0.74 linear and 0.76 with a tree, so it is a word counter, and a learned head
#: for it would be a word counter with extra steps. It is already implemented as an explicit
#: band in plugin.py, which is the honest version of the same thing and costs no labels.
#:
#: WHAT WAS CONSIDERED AND LEFT OUT, with the evidence, so it is not re-proposed:
#:
#:   `actionable` - 0.95 correlated with `elicits` here, and kappa 0.13 in the paper that
#:   named it. `hands_over` is what survives of both.
#:
#:   Cognitive demand / "how much thinking does this ask for" - kappa 0.09 to 0.34 across
#:   three independent studies. This is why `elicits` was reframed behaviourally as
#:   `hands_over`: the same construct scores 0.55-0.66 when asked as "does it hand over the
#:   next move" and 0.09-0.34 when asked as "how much thinking".
#:
#:   Uptake / revoicing - the field's flagship discourse measure, Fleiss kappa 0.286, and a
#:   single feature (student-token overlap) predicts the human labels at 0.52, so as a reward
#:   target it is gamed by parroting.
#:
#:   Anything about the student's internal state - goals, motivation, interest. Krippendorff
#:   alpha 0.03, 0.02 and 0.07, replicated across two model conditions. Below chance.
#:
#:   Which strategy the tutor should have used - three tutors pick the same action in 18% of
#:   cases, and the tutor who made a move agrees with outside raters about what it was at
#:   kappa 0.34.
#:
#:   Tone, encouragement, warmth - alpha 0.24-0.30, and rewarding it means rewarding praise,
#:   which meta-analyses at d = -0.17.
#:
#:   Holistic overall quality - two annotators agreed at r = 0.15.
NOT_RATED = ("length", "tone", "uptake", "strategy", "overall_quality")

#: WHY V2's `leak` IS WORDED DIFFERENTLY FROM THE ONE ABOVE, which is otherwise identical. The
#: old anchors were "never points at one option" at 1 and "states the CORRECT option" at 3, and a
#: rater in the pilot found the hole between them: a turn that confidently states a WRONG option
#: satisfies neither, so it falls to 2 by default. Pedagogically that is the worst case and not
#: the middle one - it removes the student's work AND misleads - so the scale was measuring the
#: wrong thing on exactly the turns that matter most. V2 asks how much of the student's remaining
#: work the turn takes away, which does not depend on whether the answer given is right; whether
#: it is right is what `correct` is for. The bug was in the first rubric from the beginning.
#:
#: Rated on the STUDENT's message, not the tutor's. Not a quality judgement and not part of
#: any reward - it is the covariate that makes the others interpretable.
#:
#: WHY THIS IS THE MOST VALUABLE THING TO ADD, despite scoring nothing. Every dimension in V2
#: is a property of the tutor's turn judged largely on its own. None asks whether the amount
#: of help matched what the student had just demonstrated - which is contingency, the defining
#: property of scaffolding (van de Pol et al. 2010) and the only property whose optimum a fixed
#: policy cannot reach, because the target moves with the student.
#:
#: And it is nearly free: `control` in that literature - open question, then hint, then tells
#: them - is the same scale as `leak`. So contingency does not need a new tutor-side code. It
#: needs the student-side one, and is then a lookup over (student_state, leak).
#:
#: COLLECTED BUT NOT REWARDED, deliberately. Tested against 85 preference judgements using
#: agent leak scores and an automatic proxy for this code, contingency agreed with the human
#: 50% of the time against leak's 44% - better, but still chance. That test cannot be trusted
#: in either direction, because the agent leak scores going into it track length at +0.59
#: against the human's +0.43. Collect this, then answer the question properly with real labels
#: before putting it in a reward.
CONTEXT: tuple[Dimension, ...] = (
    Dimension(
        "student_state",
        "What did the STUDENT's last message demonstrate? (Rate the student, not the tutor.)",
        {
            1: "Nothing to work with. No attempt, says they do not know, asks for help, or "
            "restates the question.",
            2: "An attempt. Produces a step, a number, or a claim, with no reasoning shown.",
            3: "An attempt with reasoning. Shows the step or the why that produced the answer, "
            "whether or not it is right.",
        },
    ),
)

#: Not in V2, because five dimensions already collapsed to an effective rank of 2.9 and adding
#: correlated ones makes that worse. Pilot them on a hundred turns first; keep whichever earns
#: its variance.
CANDIDATES: tuple[Dimension, ...] = (
    Dimension(
        "step_size",
        "How far does the turn ask the student to move? (Compare against the reference "
        "solution: one step is roughly one line of it.)",
        {
            1: "Too far, or nowhere. Asks for the answer, the whole method, or a leap of "
            "several lines; or asks for something the student has already done.",
            2: "Between the two, or hard to place against the reference.",
            3: "About one step, from where the student currently is.",
        },
    ),
    Dimension(
        "works_from",
        "Does the turn continue from the student's own move, or replace it with the tutor's?",
        {
            1: "Replaces it. Sets the student's reasoning aside and substitutes the tutor's "
            "method, without engaging what the student actually did.",
            2: "Neither. Does not take up the student's move, but does not override it either.",
            3: "Continues from it. Takes the student's own step, number, or claim as the "
            "starting point and pushes on it - asks what it gives, where it leads, whether it "
            "matches the question.",
        },
    ),
)

BY_KEY = {d.key: d for d in (*DIMENSIONS, *DROPPED, *DIAGNOSTIC, *V2, *CONTEXT, *CANDIDATES)}


def rubric_markdown(dimensions: tuple[Dimension, ...] = DIMENSIONS) -> str:
    """The rater-facing rubric, for whichever dimensions are being rated.

    Generated from the same objects the loader validates against, so the
    document raters read and the schema the code enforces cannot drift apart -
    which they did last time, when the rubric lived only in a hand-written file.

    ``dimensions`` exists because the default drifted anyway, in the other
    direction. A run rating the six of this round got a rubric describing the
    five of DIMENSIONS: `concise` was defined and not asked for, `length_fit`
    and `correct` were asked for and never defined. The agents guessed the two
    undefined ones from their names and the few-shot examples, and `length_fit`
    - whose whole point is that 1 and 3 are both bad - came back 2 on all 25
    holdout turns, correlating -0.36 with the human. An undefined dimension does
    not fail loudly; it comes back plausible and empty.
    """
    out = [
        "# Rubric: rating one tutor turn",
        "",
        "You see the question, the student's previous turn, and ONE tutor turn.",
        "Rate only the tutor turn. You are not judging whether the student went on",
        "to answer correctly - that is deliberately not part of this.",
        "",
        "Answer each by pointing at the text. If you find yourself forming an",
        "overall impression of the turn to decide, flag it instead of guessing:",
        "that means the question is badly written and we want to know.",
        "",
    ]
    for d in dimensions:
        out.append(f"## {d.key}: {d.lo}-{d.hi} — {d.question}")
        out.append("")
        out.extend(f"- **{score}** — {text}" for score, text in sorted(d.anchors.items()))
        out.append("")
    out += [
        "## flag",
        "",
        "Set `flag` to a short string on any turn where the rubric did not fit,",
        "instead of forcing a score. A dimension that gets flagged often is a",
        "dimension to rewrite or drop, and that is worth more than a guessed number.",
    ]
    return "\n".join(out)


def validate(record: dict, dimensions: tuple[Dimension, ...] = DIMENSIONS) -> list[str]:
    """Problems with one rater's record, as a list of human-readable strings.

    ``dimensions`` narrows the check, for the case where only some are being
    re-rated and the record is not meant to be complete.
    """
    problems = []
    for d in dimensions:
        if d.key not in record:
            if not record.get("flag"):
                problems.append(f"missing {d.key}")
            continue
        value = record[d.key]
        if not isinstance(value, int) or not (d.lo <= value <= d.hi):
            problems.append(f"{d.key}={value!r} outside {d.lo}-{d.hi}")
    return problems


if __name__ == "__main__":
    print(rubric_markdown())
