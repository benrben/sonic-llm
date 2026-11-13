Incident report INC-7955

Incident report INC-7955 for the Open Gate project of Vantage Logistics. It describes what failed, who was affected and what had changed.

Teams

Tannery team is led by Vera Pellegrini. Its members are Clara Whitlock and Oskar Yilmaz.
Quarry team is led by Nils Pellegrini. Its members are Anouk Whitlock and Ilse Brandt.
Harbour team is led by Yara Sandoval. Its members are Esme Haddad and Celia Lindqvist.

Components

| Code | Component | Quantity | Unit price (euros) | Due | Status |
|---|---|---|---|---|---|
| VN-2314 | heavy pipe clamp | 50 | 323 | 8 January 2027 | monitoring |
| VN-2341 | spare signal lamp | 147 | 403 | 22 May 2027 | monitoring |
| VN-1980 | spare sensor pack | 149 | 190 | 6 February 2027 | monitoring |
| VN-5401 | copper bracket kit | 122 | 31 | 3 February 2027 | fixed |
| VN-1046 | coated battery rack | 158 | 108 | 16 January 2027 | open |
| VN-8665 | spare pipe clamp | 105 | 328 | 20 March 2027 | monitoring |

Budget

The approved budget is 38,700 euros (42,570 US dollars at the finance rate of 1.10).

On 31 March 2027 the service stopped answering for 54 minutes. 446 customers were affected. The outage was caused by change CR-1662.

Changes

Change CR-1662 was approved by Marta Marchetti on 10 January 2027. Change CR-9177 was approved by Olga Brandt on 9 January 2027.

Revisions

Revision: the due date of VN-1980 was revised to 13 February 2027.

Rating by customers affected: minor when it is at most 150; major when it is above 150 and at most 250; critical when it is above 250.

Appendix A: contact list

| Person | Team | Desk |
|---|---|---|
| Vera Pellegrini | Tannery team | 4200 |
| Clara Whitlock | Quarry team | 3690 |
| Oskar Yilmaz | Harbour team | 5332 |
| Nils Pellegrini | Tannery team | 6344 |
| Anouk Whitlock | Quarry team | 3343 |
| Ilse Brandt | Harbour team | 4938 |

Erratum

Erratum to the table: the unit price of VN-5401 is 171 euros, not 31 euros.

Earlier incidents. A similar problem occurred some months ago and was recorded as ticket RF-113. At that time the cause was different, but the symptoms were the same, and the runbook did not tell the engineer where to look. We should have connected the two events earlier, and the review will look at how to make older reports easier to find.

Communication. The status page was updated three times, and each update was reviewed by Celia Marchetti before it was published. An internal message went to the wider team, and a short note was sent to customers who had contacted support. We believe the updates were honest and clear, although one of them used a technical term that customers did not know.

Metrics. The service normally answers in about 58 milliseconds and fails fewer than 91 requests in a thousand. During the incident the failure rate rose sharply and the average answer time went up more than tenfold. Within 81 minutes of the fix being applied, both numbers returned to normal values. Graphs of the period are attached to the ticket.
