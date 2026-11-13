Incident report INC-5605

Incident report INC-5605 for the Blue Pantry project of Ironbark Logistics. It describes what failed, who was affected and what had changed.

Teams

Bridge team is led by Nadia Oyelaran. Its members are Marta Salazar and Tomas Whitlock.
Harbour team is led by Olga Oyelaran. Its members are Celia Novak and Hugo Ulbricht.
Beacon team is led by Bruno Kowalczyk. Its members are Rosa Tavares and Selma Rasmussen.

Components

| Code | Component | Quantity | Unit price (euros) | Due | Status |
|---|---|---|---|---|---|
| VN-2030 | compact pipe clamp | 136 | 24 | 11 February 2027 | monitoring |
| VN-2003 | copper relay board | 95 | 402 | 7 January 2027 | fixed |
| VN-1199 | coated gauge set | 157 | 117 | 24 March 2027 | monitoring |
| VN-8749 | spare bracket kit | 73 | 48 | 14 January 2027 | monitoring |
| VN-1079 | coated fan module | 98 | 219 | 15 February 2027 | fixed |
| VN-1792 | copper signal lamp | 77 | 192 | 25 February 2027 | open |

Budget

The approved budget is 25,500 euros (28,050 US dollars at the finance rate of 1.10).

On 19 April 2027 the service stopped answering for 64 minutes. 1,290 customers were affected. The outage was caused by change CR-6680.

Changes

Change CR-6680 was approved by Rosa Vasquez on 15 January 2027. Change CR-4188 was approved by Sven Fontaine on 5 February 2027.

Revisions

Revision: the due date of VN-1199 was revised to 16 April 2027.

Rating by customers affected: minor when it is at most 750; major when it is above 750 and at most 1,700; critical when it is above 1,700.

Appendix A: contact list

| Person | Team | Desk |
|---|---|---|
| Nadia Oyelaran | Bridge team | 6710 |
| Marta Salazar | Harbour team | 7170 |
| Tomas Whitlock | Beacon team | 5164 |
| Olga Oyelaran | Bridge team | 5725 |
| Celia Novak | Harbour team | 7115 |
| Hugo Ulbricht | Beacon team | 7063 |

Erratum

Erratum to the table: the unit price of VN-8749 is 146 euros, not 48 euros.

Lessons. Small changes can cause large problems when the checks are skipped. People who are tired make fewer good decisions, so the on-call rota will be changed so that nobody is the first responder for more than 85 nights in a row. Alerts should say what to do, and not only what is wrong, and every important service needs a named owner.

Costs. The incident used about 85 hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.

Metrics. The service normally answers in about 85 milliseconds and fails fewer than 23 requests in a thousand. During the incident the failure rate rose sharply and the average answer time went up more than tenfold. Within 53 minutes of the fix being applied, both numbers returned to normal values. Graphs of the period are attached to the ticket.
