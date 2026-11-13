Incident report INC-5034

Incident report INC-5034 for the Green Ledger project of Copperleaf Foods. It describes what failed, who was affected and what had changed.

Teams

Willow team is led by Wim Pellegrini. Its members are Rosa Sandoval and Rosa Olsen.
Meadow team is led by Pavel Pellegrini. Its members are Viktor Petrov and Celia Whitlock.
Bridge team is led by Yara Vasquez. Its members are Dmitri Nakagawa and Nils Thorne.
Beacon team is led by Wim Holloway. Its members are Nils Fontaine and Mira Tennant.
Harbour team is led by Zoran Pellegrini. Its members are Irene Varga and Elena Yilmaz.

Components

| Code | Component | Quantity | Unit price (euros) | Due | Status |
|---|---|---|---|---|---|
| PM-6005 | sealed cable drum | 79 | 428 | February 4, 2027 | monitoring |
| PM-6050 | steel pipe clamp | 47 | 341 | January 15, 2027 | fixed |
| PM-6779 | copper valve set | 11 | 159 | May 9, 2027 | open |
| PM-2068 | spare pipe clamp | 107 | 371 | April 20, 2027 | fixed |
| PM-3595 | compact gauge set | 113 | 20 | January 14, 2027 | fixed |
| PM-9400 | quiet signal lamp | 118 | 331 | April 8, 2027 | fixed |
| PM-8642 | quiet pump unit | 5 | 425 | May 6, 2027 | fixed |
| PM-1967 | steel pump unit | 55 | 273 | May 11, 2027 | monitoring |
| PM-3891 | coated valve set | 61 | 28 | January 28, 2027 | fixed |
| PM-7066 | sealed signal lamp | 99 | 122 | January 3, 2027 | monitoring |
| PM-1767 | copper filter bank | 56 | 60 | March 29, 2027 | fixed |
| PM-9782 | coated bracket kit | 63 | 23 | April 15, 2027 | monitoring |
| PM-2904 | spare battery rack | 120 | 165 | March 9, 2027 | fixed |
| PM-8526 | quiet gasket roll | 17 | 308 | May 20, 2027 | fixed |
| PM-7694 | coated cable drum | 146 | 473 | May 1, 2027 | fixed |
| PM-9581 | spare signal lamp | 24 | 326 | May 5, 2027 | fixed |

Budget

The approved budget is 14,700 euros (16,170 US dollars at the finance rate of 1.10).

On April 17, 2027 the service stopped answering for 37 minutes. 3,806 customers were affected. The outage was caused by change CR-3885.

Revisions

Revision: the due date of PM-2068 was revised to April 30, 2027.

Rating by customers affected: minor when it is at most 4,950; major when it is above 4,950 and at most 7,250; critical when it is above 7,250.

Appendix A: contact list

| Person | Team | Desk |
|---|---|---|
| Wim Pellegrini | Willow team | 3022 |
| Rosa Sandoval | Meadow team | 5609 |
| Rosa Olsen | Bridge team | 5623 |
| Pavel Pellegrini | Beacon team | 4242 |
| Viktor Petrov | Harbour team | 8437 |
| Celia Whitlock | Willow team | 6179 |

Erratum

Erratum to the table: the unit price of PM-3595 is 234 euros, not 20 euros.

Costs. The incident used about 92 hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.

Lessons. Small changes can cause large problems when the checks are skipped. People who are tired make fewer good decisions, so the on-call rota will be changed so that nobody is the first responder for more than 92 nights in a row. Alerts should say what to do, and not only what is wrong, and every important service needs a named owner.

Earlier incidents. A similar problem occurred some months ago and was recorded as ticket RF-887. At that time the cause was different, but the symptoms were the same, and the runbook did not tell the engineer where to look. We should have connected the two events earlier, and the review will look at how to make older reports easier to find.
