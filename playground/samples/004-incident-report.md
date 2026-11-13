Incident report INC-3593

Incident report INC-3593 for the Bright Loom project of Northwind Foods. It describes what failed, who was affected and what had changed.

Teams

Anchor team is led by Marta Ulbricht. Its members are Tilda Petrov and Nils Nakagawa.
Bridge team is led by Felix Ulbricht. Its members are Nils Quinlan and Yara Castellan.
Foundry team is led by Ilse Varga. Its members are Tomas Tavares and Mira Eriksen.
Willow team is led by Selma Ulbricht. Its members are Anouk Zielinski and Leon Castellan.
Meadow team is led by Sven Olsen. Its members are Leon Lindqvist and Petra Castellan.

Components

| Code | Component | Quantity | Unit price (euros) | Due | Status |
|---|---|---|---|---|---|
| KX-5230 | rapid cable drum | 5 | 346 | January 23, 2027 | fixed |
| KX-5203 | copper pipe clamp | 80 | 20 | January 22, 2027 | monitoring |
| KX-9436 | heavy bracket kit | 81 | 434 | March 28, 2027 | open |
| KX-1600 | modular gauge set | 117 | 260 | April 11, 2027 | fixed |
| KX-3084 | coated cable drum | 108 | 355 | January 6, 2027 | monitoring |
| KX-4941 | heavy gauge set | 131 | 306 | April 20, 2027 | monitoring |
| KX-6809 | compact pump unit | 160 | 211 | March 23, 2027 | monitoring |
| KX-2681 | sealed panel frame | 75 | 121 | May 4, 2027 | monitoring |
| KX-9007 | steel pump unit | 112 | 213 | April 5, 2027 | monitoring |
| KX-1321 | rapid signal lamp | 78 | 324 | May 27, 2027 | fixed |
| KX-9727 | sealed relay board | 38 | 442 | May 21, 2027 | fixed |
| KX-9776 | copper panel frame | 94 | 287 | April 29, 2027 | monitoring |
| KX-9533 | outdoor fan module | 119 | 414 | April 28, 2027 | monitoring |
| KX-7446 | steel sensor pack | 136 | 479 | February 16, 2027 | fixed |
| KX-1312 | sealed cable drum | 122 | 339 | January 7, 2027 | monitoring |
| KX-2701 | modular bracket kit | 127 | 208 | March 16, 2027 | monitoring |

Budget

The approved budget is 47,000 euros (51,700 US dollars at the finance rate of 1.10).

On March 10, 2027 the service stopped answering for 161 minutes. 6,556 customers were affected. The outage was caused by change CR-1764.

Revisions

Revision: the due date of KX-1600 was revised to May 19, 2027.

Rating by customers affected: minor when it is at most 8,500; major when it is above 8,500 and at most 12,450; critical when it is above 12,450.

Appendix A: contact list

| Person | Team | Desk |
|---|---|---|
| Marta Ulbricht | Anchor team | 5652 |
| Tilda Petrov | Bridge team | 4827 |
| Nils Nakagawa | Foundry team | 7427 |
| Felix Ulbricht | Willow team | 6467 |
| Nils Quinlan | Meadow team | 7952 |
| Yara Castellan | Anchor team | 6334 |

Erratum

Erratum to the table: the unit price of KX-3084 is 30 euros, not 355 euros.

Costs. The incident used about 41 hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.

Metrics. The service normally answers in about 41 milliseconds and fails fewer than 76 requests in a thousand. During the incident the failure rate rose sharply and the average answer time went up more than tenfold. Within 39 minutes of the fix being applied, both numbers returned to normal values. Graphs of the period are attached to the ticket.

Communication. The status page was updated three times, and each update was reviewed by Pavel Novak before it was published. An internal message went to the wider team, and a short note was sent to customers who had contacted support. We believe the updates were honest and clear, although one of them used a technical term that customers did not know.
