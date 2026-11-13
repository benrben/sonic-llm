Incident report INC-7046

Incident report INC-7046 for the Iron Orchard project of Halcyon Engineering. It describes what failed, who was affected and what had changed.

Teams

Anchor team is led by Nadia Petrov. Its members are Sven Salazar and Sven Rasmussen.
Harbour team is led by Tilda Petrov. Its members are Rafael Dvorak and Felix Holloway.
Tannery team is led by Sven Oyelaran. Its members are Zoran Castellan and Selma Grasso.

Components

| Code | Component | Quantity | Unit price (euros) | Due | Status |
|---|---|---|---|---|---|
| KX-1421 | coated filter bank | 88 | 401 | May 10, 2027 | monitoring |
| KX-1412 | compact filter bank | 155 | 315 | January 27, 2027 | monitoring |
| KX-9524 | quiet signal lamp | 160 | 232 | April 18, 2027 | monitoring |
| KX-9642 | coated sensor pack | 55 | 264 | February 11, 2027 | monitoring |
| KX-7004 | spare pipe clamp | 153 | 253 | March 11, 2027 | monitoring |
| KX-8629 | rapid valve set | 58 | 61 | February 19, 2027 | open |

Budget

The approved budget is 31,200 euros (34,320 US dollars at the finance rate of 1.10).

On April 9, 2027 the service stopped answering for 122 minutes. 115 customers were affected. The outage was caused by change CR-5950.

Changes

Change CR-5950 was approved by Clara Jankowski on March 19, 2027. Change CR-2583 was approved by Tomas Ulbricht on February 23, 2027.

Revisions

Revision: the due date of KX-9524 was revised to May 11, 2027.

Rating by customers affected: minor when it is at most 50; major when it is above 50 and at most 100; critical when it is above 100.

Appendix A: contact list

| Person | Team | Desk |
|---|---|---|
| Nadia Petrov | Anchor team | 7811 |
| Sven Salazar | Harbour team | 7805 |
| Sven Rasmussen | Tannery team | 6457 |
| Tilda Petrov | Anchor team | 3811 |
| Rafael Dvorak | Harbour team | 5912 |
| Felix Holloway | Tannery team | 4501 |

Erratum

Erratum to the table: the unit price of KX-9642 is 271 euros, not 264 euros.

Lessons. Small changes can cause large problems when the checks are skipped. People who are tired make fewer good decisions, so the on-call rota will be changed so that nobody is the first responder for more than 7 nights in a row. Alerts should say what to do, and not only what is wrong, and every important service needs a named owner.

Earlier incidents. A similar problem occurred some months ago and was recorded as ticket RF-112. At that time the cause was different, but the symptoms were the same, and the runbook did not tell the engineer where to look. We should have connected the two events earlier, and the review will look at how to make older reports easier to find.

Costs. The incident used about 7 hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.
