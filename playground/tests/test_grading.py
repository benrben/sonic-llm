"""The grading rules (grading.py): what counts as the right answer, and what does not."""

import pytest

from sonic_playground import grading
from sonic_playground.grading import CORRECT, EMPTY, WRONG, grade_answer, grade_fields, grade_label

MONEY = ["1,250 euros", "1,250", "1250 euros", "1250"]
DATE = ["14 March 2027", "March 14, 2027", "2027-03-14"]


def test_the_right_answer_inside_a_sentence_counts():
    assert grade_answer("The approved budget is 1,250 euros, to be spent against receipts.", MONEY) == CORRECT
    assert grade_answer("**1250**", MONEY) == CORRECT
    assert grade_answer("It was approved at €1,250.", MONEY) == CORRECT
    assert grade_answer("Marta Okonkwo", ["Marta Okonkwo", "Okonkwo"]) == CORRECT
    assert grade_answer("the CHAIR was ms. okonkwo!", ["Marta Okonkwo", "Okonkwo"]) == CORRECT


@pytest.mark.parametrize("reply", ["12500 euros", "It is 11,250 euros", "1250.5 euros", "about 125", "Q1250", "I do not know"])
def test_a_wrong_number_that_contains_the_right_digits_is_wrong(reply):
    assert grade_answer(reply, MONEY) == WRONG


def test_two_different_answers_of_the_same_kind_are_hedging_and_fail():
    assert grade_answer("1,250 euros or 1,520 euros", MONEY) == WRONG
    assert grade_answer("Either 1,520 or 1,250.", MONEY) == WRONG
    assert grade_answer("1250, not 1520", MONEY) == WRONG
    assert grade_answer("It is 14 March 2027, or maybe 21 March 2027.", DATE) == WRONG
    assert grade_answer("INC-4821 or INC-4822", ["INC-4821"]) == WRONG


def test_other_numbers_that_are_not_offered_as_the_answer_do_not_count_as_hedging():
    assert grade_answer("The total is 1,250 euros for 3 items, delivered to room 4.", MONEY) == CORRECT
    assert grade_answer("14 March 2027, about 12 members", DATE) == CORRECT


@pytest.mark.parametrize("reply", ["14 March 2027", "March 14, 2027", "march 14th, 2027", "14th of March 2027", "14 Mar 2027", "2027-03-14",
                                   "On the 14 March, 2027 at nine"])
def test_dates_are_the_same_in_every_common_form(reply):
    assert grade_answer(reply, DATE[:1]) == CORRECT


def test_a_date_with_another_day_is_wrong():
    assert grade_answer("15 March 2027", DATE) == WRONG
    assert grade_answer("14 March 2026", DATE) == WRONG


def test_nothing_after_the_thinking_is_an_empty_answer_and_thinking_is_never_graded():
    assert grade_answer("", MONEY) == EMPTY and grade_answer("  \n", MONEY) == EMPTY and grade_answer("...", MONEY) == EMPTY
    assert grade_answer("<think>maybe 1250 euros</think>", MONEY) == EMPTY
    assert grade_answer("<think>maybe 1250 euros</think>I cannot tell.", MONEY) == WRONG
    assert grade_answer("<think>the answer is 1520</think>1,250 euros", MONEY) == CORRECT
    assert grade_answer("I think it is 1250 euros</think>Done", MONEY) == WRONG  # a template that opens the thinking itself
    assert grade_answer("<think>still thinking about 1250 euros", MONEY) == EMPTY


def test_a_longer_word_or_code_is_not_the_name_or_code():
    assert grade_answer("KX-1961", ["KX-196"]) == WRONG
    assert grade_answer("Rosa Thornedale", ["Rosa Thorne", "Thorne"]) == WRONG
    assert grade_answer("The model is KX-196.", ["KX-196"]) == CORRECT


def test_normalize_handles_quotes_articles_and_separators():
    assert grading.tokens("“The Harbour Library”, 1,250,000 units") == ["the", "harbour", "library", "1250000", "units"]
    assert grade_answer("It is 'Hotel Aurelia'.", ["The Hotel Aurelia"]) == CORRECT


LABELS = ["meeting notes", "recipe", "invoice", "lab report", "incident report"]


def test_a_label_must_be_the_only_one_named_and_the_right_one():
    assert grade_label("recipe", LABELS, "recipe") == CORRECT
    assert grade_label("This document is an Invoice.", LABELS, "invoice") == CORRECT
    assert grade_label("**Incident report**", LABELS, "incident report") == CORRECT
    assert grade_label("invoice or recipe", LABELS, "invoice") == WRONG  # two labels: hedging
    assert grade_label("recipe", LABELS, "invoice") == WRONG
    assert grade_label("something else", LABELS, "invoice") == WRONG
    assert grade_label("<think>invoice</think>", LABELS, "invoice") == EMPTY


FIELDS = {"chair": "Rosa Thorne", "meeting_date": "14 January 2027", "approved_budget_eur": "1465"}


def test_extraction_compares_every_field_and_a_reply_that_is_not_json_fails_them_all():
    ok = '{"chair": "Rosa Thorne", "meeting_date": "2027-01-14", "approved_budget_eur": 1465}'
    assert grade_fields(ok, FIELDS) == (CORRECT, {"chair": True, "meeting_date": True, "approved_budget_eur": True})
    assert grade_fields("```json\n" + ok + "\n```", FIELDS)[0] == CORRECT
    verdict, checks = grade_fields('{"chair": "Rosa Thorne", "meeting date": "January 15, 2027", "approved_budget_eur": "1,465 euros"}', FIELDS)
    assert verdict == WRONG and checks == {"chair": True, "meeting_date": False, "approved_budget_eur": True}
    assert grade_fields('Here it is: {"chair": "Rosa Thorne"}', FIELDS) == (WRONG, {k: False for k in FIELDS})
    assert grade_fields('{"chair": "Rosa Thorne"', FIELDS)[0] == WRONG  # cut off
    assert grade_fields("", FIELDS)[0] == EMPTY
    assert grade_fields('{"chair": null, "meeting_date": ["14 January 2027"]}', FIELDS)[1] == {"chair": False, "meeting_date": True, "approved_budget_eur": False}


# -- questions of the sample key: types, tolerance, NOT STATED and the reasons ------------------------------------------------------
def money(n, **more):
    return {"accept": [f"{n:,} euros", f"{n:,}"], "type": "number", "skill": "arithmetic", **more}


@pytest.mark.parametrize("reply", ["3400", "3,400", "€3,400", "EUR 3400.00", "3,400.00 euros", "**3,400 euros**", "3400 €"])
def test_a_number_is_right_with_or_without_separators_currency_and_zero_decimals(reply):
    assert grading.grade_question(reply, money(3400)) == (CORRECT, "")


@pytest.mark.parametrize("reply, reason", [("34000", "wrong arithmetic"), ("3,401 euros", "wrong arithmetic"), ("3,400 or 3,500 euros", "gave two answers"),
                                           ("I could not say", "wrong answer"), ("", "no answer"), ("<think>3400</think>", "no answer")])
def test_a_wrong_number_says_why(reply, reason):
    verdict, why = grading.grade_question(reply, money(3400))
    assert (verdict, why) == (EMPTY if reason == "no answer" else WRONG, reason)


def test_the_trap_a_wrong_answer_fell_into_is_the_reason():
    old = money(3400, reject=["2,975 euros", "2975"], trap="old")
    assert grading.grade_question("2,975 euros", old) == (WRONG, "gave the old value")
    usd = {"accept": ["5,500 dollars", "5,500"], "type": "number", "skill": "lookup", "reject": ["5,000"], "trap": "currency"}
    assert grading.grade_question("5000 euros", usd) == (WRONG, "gave the amount in the other currency")
    twin = {"accept": ["Ilse Brandt"], "reject": ["Tomas Brandt"], "trap": "similar", "skill": "lookup"}
    assert grading.grade_question("Tomas Brandt", twin) == (WRONG, "gave the look-alike")
    assert grading.grade_question("Ilse Brandt", twin) == (CORRECT, "")
    assert grading.grade_question("Nils Olsen", twin) == (WRONG, "wrong answer")


PERCENT = {"accept": ["37.5"], "type": "percent", "tol": 0.5, "skill": "arithmetic"}


@pytest.mark.parametrize("reply, verdict", [("37.5", CORRECT), ("37.5%", CORRECT), ("about 37.9 %", CORRECT), ("37", CORRECT), ("38", CORRECT), ("37.0", CORRECT),
                                            ("38.1", WRONG), ("36.9", WRONG), ("37 or 40", WRONG), ("", EMPTY)])
def test_a_percentage_is_right_within_half_a_point_of_the_key(reply, verdict):
    assert grading.grade_question(reply, PERCENT)[0] == verdict


def test_the_tolerance_is_only_for_percentages():
    assert grading.grade_question("3401", money(3400))[0] == WRONG
    assert grading.grade_question("17 days", {"accept": ["18 days", "18"], "type": "days", "skill": "date-math"})[0] == WRONG


WEEKDAY = {"accept": ["Tuesday", "Tue"], "type": "weekday", "skill": "date-math"}


def test_a_weekday_is_the_name_or_its_short_form_and_another_day_is_wrong_date_math():
    assert [grading.grade_question(r, WEEKDAY)[0] for r in ("Tuesday", "tuesday.", "It is a Tue", "**Tuesday**")] == [CORRECT] * 4
    assert grading.grade_question("Wednesday", WEEKDAY) == (WRONG, "wrong date math")
    assert grading.grade_question("Tuesday or Wednesday", WEEKDAY) == (WRONG, "gave two answers")


DAYS = {"accept": ["17 days", "17", "seventeen"], "type": "days", "skill": "date-math"}


@pytest.mark.parametrize("reply", ["17", "17 days", "seventeen days", "Seventeen", "2 weeks and 3 days", "2 weeks, 3 days", "two weeks and three days",
                                   "17 days (2 weeks and 3 days)"])
def test_a_duration_is_the_same_in_days_weeks_and_words(reply):
    assert grading.grade_question(reply, DAYS)[0] == CORRECT


@pytest.mark.parametrize("reply", ["18 days", "2 weeks", "2 weeks and 4 days", "three weeks", "17 or 18 days", "twenty-one days"])
def test_another_duration_is_wrong(reply):
    assert grading.grade_question(reply, DAYS)[0] == WRONG


def test_durations_are_only_converted_for_questions_that_ask_for_one():
    assert grading.grade_question("2 weeks and 3 days", {"accept": ["17"], "type": "number", "skill": "arithmetic"})[0] == WRONG
    assert grading._spell_durations("twenty-one days") == "21 days" and grading._spell_durations("one week") == "7 days"


NOT_STATED = {"accept": ["NOT STATED"], "type": "nostated", "skill": "negative"}


@pytest.mark.parametrize("reply", ["NOT STATED", "not stated", "Not stated.", "**NOT STATED**", "NOT STATED: the document does not say who runs it",
                                   "<think>maybe 40</think>NOT STATED"])
def test_not_stated_passes_when_the_reply_says_it_and_gives_no_value(reply):
    assert grading.grade_question(reply, NOT_STATED) == (CORRECT, "")


@pytest.mark.parametrize("reply, reason", [("Ilse Brandt", "invented an answer"), ("040 555 1234", "invented an answer"), ("NOT STATED, but probably 12 days", "invented an answer"),
                                           ("NOT STATED " + "word " * 20, "invented an answer"), ("The document does not mention it.", "did not say NOT STATED"),
                                           ("I don't know", "did not say NOT STATED"), ("", "no answer")])
def test_not_stated_fails_when_a_value_is_invented_or_the_phrase_is_missing(reply, reason):
    verdict, why = grading.grade_question(reply, NOT_STATED)
    assert (verdict, why) == (EMPTY if not reply else WRONG, reason)


def test_a_value_does_not_pass_a_question_that_has_one_just_because_it_says_not_stated():
    assert grading.grade_question("NOT STATED", money(3400)) == (WRONG, "wrong answer")


def test_hedging_still_fails_in_every_type():
    date = {"accept": ["14 March 2027", "2027-03-14"], "type": "date", "skill": "update", "reject": ["2 May 2027"], "trap": "old"}
    assert grading.grade_question("14 March 2027", date)[0] == CORRECT
    assert grading.grade_question("14 March 2027, or maybe 2 May 2027", date) == (WRONG, "gave two answers")
    assert grading.grade_question("2 May 2027", date) == (WRONG, "gave the old value")
    assert grading.grade_question("It could be 40 or 41 percent", PERCENT)[0] == WRONG
