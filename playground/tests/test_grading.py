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
