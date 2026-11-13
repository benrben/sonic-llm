# Weekly status update: customer portal redesign, week 11

**Prepared by:** Lena Fischer, project lead
**Period:** Monday 11 March to Friday 15 March
**Overall status:** amber. The scope is on track but the launch date is at risk.

## Summary

The new login and account pages passed internal review this week. The invoice pages are two days behind plan because the billing
service returned inconsistent totals in the test system. The target launch is still Monday 8 April, but if the billing issue is not
fixed by Friday 22 March we will propose moving to Monday 22 April.

## Done this week

- Login, password reset and two-factor setup pages finished and approved by the security team (Sven Aaltonen).
- Account overview page built; it loads in 1.1 seconds on the staging server, down from 3.4 seconds on the old portal.
- Translation files delivered for German and Spanish; the French file is still outstanding.
- 38 of 44 accessibility issues from the audit were fixed; the remaining six concern colour contrast and are scheduled for next week.
- Usability test with eight customers held on Wednesday. Seven of eight completed the task of finding last month's invoice in
  under a minute. The eighth did not notice the new menu, so we will make it more visible.

## In progress

- Invoice list and invoice detail pages (Hugo Marchetti): 60 percent complete.
- Payment method page: waiting for the design review on Monday.
- Data migration script for old customer preferences: first dry run on 18 March.

## Risks and issues

| Risk | Impact | What we are doing | Owner |
|------|--------|-------------------|-------|
| Billing service totals differ from the old system by cents on 2 percent of invoices | high | Billing team is checking the rounding rules | Priya Raman |
| French translations are late | medium | Agency promised delivery by Wednesday 20 March | Lena |
| Only one engineer knows the legacy export | medium | Pairing session planned, and the export is being documented | Hugo |
| Marketing wants a banner on launch day | low | Agreed to add after the main release, no scope change | Lena |

## Numbers

- Budget: 212,000 euros planned for the project; 141,500 euros spent so far, which is 67 percent of the budget with the project at
  about 65 percent of scope.
- Open bugs: 27, of which 3 are critical and all three are in the invoice pages.
- Test coverage of the new front end: 78 percent, with a target of 85 percent by launch.

## Decisions needed

1. Approve the fallback launch date of 22 April if the billing issue is not fixed by 22 March (sponsor, Ines Duarte, by Wednesday).
2. Approve two extra days of contractor time to finish the accessibility fixes (about 1,600 euros).

## Next week

Finish the invoice pages, run the first migration dry run, fix the six remaining accessibility issues, and hold the second round of
usability testing with five business customers on Thursday.
