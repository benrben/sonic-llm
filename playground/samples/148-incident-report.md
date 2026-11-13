Incident report INC-2774

Incident report INC-2774 for the Blue Pantry project of Halcyon Dairy. It describes what failed, who was affected and what had changed.

Teams

Bridge team is led by Wim Varga. Its members are Ulrich Kessler and Hugo Sandoval.
Granary team is led by Celia Varga. Its members are Bastian Oyelaran and Hugo Laurent.
Foundry team is led by Viktor Lindqvist. Its members are Mira Iversen and Tomas Okonkwo.
Quarry team is led by Nadia Thorne. Its members are Felix Laurent and Vera Novak.
Beacon team is led by Dmitri Eriksen. Its members are Dmitri Rossetti and Ilse Novak.

Components

| Code | Component | Quantity | Unit price (euros) | Due | Status |
|---|---|---|---|---|---|
| KX-5969 | steel pipe clamp | 95 | 228 | 26 February 2027 | monitoring |
| KX-5996 | spare pipe clamp | 41 | 433 | 21 January 2027 | fixed |
| KX-6642 | sealed signal lamp | 128 | 427 | 29 April 2027 | monitoring |
| KX-7223 | coated fan module | 30 | 253 | 25 January 2027 | fixed |
| KX-6225 | coated sensor pack | 137 | 437 | 4 February 2027 | fixed |
| KX-1399 | copper filter bank | 9 | 413 | 13 January 2027 | monitoring |
| KX-7190 | coated pump unit | 126 | 277 | 19 January 2027 | fixed |
| KX-6930 | quiet gasket roll | 159 | 416 | 26 April 2027 | open |
| KX-1662 | rapid bracket kit | 11 | 67 | 21 May 2027 | fixed |
| KX-3059 | heavy filter bank | 100 | 52 | 25 May 2027 | fixed |
| KX-6365 | bonded filter bank | 125 | 394 | 6 May 2027 | fixed |
| KX-4248 | sealed filter bank | 59 | 456 | 30 May 2027 | monitoring |
| KX-2314 | outdoor relay board | 107 | 266 | 15 March 2027 | fixed |
| KX-1705 | compact panel frame | 134 | 479 | 31 March 2027 | monitoring |
| KX-9764 | compact pipe clamp | 33 | 250 | 4 January 2027 | fixed |
| KX-2700 | steel bracket kit | 110 | 472 | 14 January 2027 | fixed |

Budget

The approved budget is 46,200 euros (50,820 US dollars at the finance rate of 1.10).

On 8 May 2027 the service stopped answering for 13 minutes. 4,818 customers were affected. The outage was caused by change CR-4549.

Revisions

Revision: the due date of KX-6642 was revised to 18 May 2027.

Rating by customers affected: minor when it is at most 6,250; major when it is above 6,250 and at most 9,150; critical when it is above 9,150.

Appendix A: contact list

| Person | Team | Desk |
|---|---|---|
| Wim Varga | Bridge team | 8006 |
| Ulrich Kessler | Granary team | 4943 |
| Hugo Sandoval | Foundry team | 5414 |
| Celia Varga | Quarry team | 7060 |
| Bastian Oyelaran | Beacon team | 5849 |
| Hugo Laurent | Bridge team | 3239 |

Erratum

Erratum to the table: the unit price of KX-7223 is 303 euros, not 253 euros.

Earlier incidents. A similar problem occurred some months ago and was recorded as ticket RF-757. At that time the cause was different, but the symptoms were the same, and the runbook did not tell the engineer where to look. We should have connected the two events earlier, and the review will look at how to make older reports easier to find.

Costs. The incident used about 24 hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.

Lessons. Small changes can cause large problems when the checks are skipped. People who are tired make fewer good decisions, so the on-call rota will be changed so that nobody is the first responder for more than 24 nights in a row. Alerts should say what to do, and not only what is wrong, and every important service needs a named owner.
