Incident report INC-5906

Incident report INC-5906 for the Slow River project of Quillon Supply. It describes what failed, who was affected and what had changed.

Teams

Foundry team is led by Anouk Olsen. Its members are Tomas Petrov and Oskar Sandoval.
Tannery team is led by Olga Olsen. Its members are Sven Iversen and Selma Laurent.
Willow team is led by Viktor Abernethy. Its members are Oskar Brandt and Wim Petrov.
Meadow team is led by Hugo Varga. Its members are Rosa Laurent and Dmitri Rasmussen.
Quarry team is led by Celia Sandoval. Its members are Tomas Rasmussen and Bruno Thorne.

Components

| Code | Component | Quantity | Unit price (euros) | Due | Status |
|---|---|---|---|---|---|
| RT-3306 | coated relay board | 80 | 414 | January 6, 2027 | monitoring |
| RT-3360 | spare signal lamp | 90 | 194 | March 2, 2027 | fixed |
| RT-6874 | compact valve set | 122 | 146 | May 19, 2027 | fixed |
| RT-3025 | modular bracket kit | 32 | 170 | March 20, 2027 | fixed |
| RT-7649 | sealed battery rack | 93 | 257 | February 4, 2027 | monitoring |
| RT-3821 | outdoor relay board | 109 | 463 | February 17, 2027 | fixed |
| RT-6480 | bonded sensor pack | 144 | 420 | April 5, 2027 | monitoring |
| RT-6840 | quiet fan module | 36 | 415 | March 26, 2027 | fixed |
| RT-1986 | modular battery rack | 131 | 124 | February 20, 2027 | fixed |
| RT-6820 | modular filter bank | 100 | 91 | March 16, 2027 | fixed |
| RT-2619 | sealed filter bank | 17 | 478 | May 24, 2027 | monitoring |
| RT-6958 | spare sensor pack | 155 | 294 | April 14, 2027 | monitoring |
| RT-3359 | sealed pipe clamp | 51 | 111 | January 5, 2027 | fixed |
| RT-4220 | compact relay board | 73 | 24 | January 14, 2027 | open |
| RT-8642 | bonded valve set | 66 | 134 | February 9, 2027 | fixed |
| RT-7796 | compact sensor pack | 10 | 389 | January 16, 2027 | monitoring |

Budget

The approved budget is 45,400 euros (49,940 US dollars at the finance rate of 1.10).

On April 22, 2027 the service stopped answering for 114 minutes. 2,370 customers were affected. The outage was caused by change CR-4887.

Revisions

Revision: the due date of RT-6874 was revised to June 4, 2027.

Rating by customers affected: minor when it is at most 700; major when it is above 700 and at most 1,400; critical when it is above 1,400.

Appendix A: contact list

| Person | Team | Desk |
|---|---|---|
| Anouk Olsen | Foundry team | 5596 |
| Tomas Petrov | Tannery team | 4902 |
| Oskar Sandoval | Willow team | 8681 |
| Olga Olsen | Meadow team | 8227 |
| Sven Iversen | Quarry team | 6629 |
| Selma Laurent | Foundry team | 4478 |

Erratum

Erratum to the table: the unit price of RT-3025 is 148 euros, not 170 euros.

Costs. The incident used about 15 hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.

Earlier incidents. A similar problem occurred some months ago and was recorded as ticket RF-266. At that time the cause was different, but the symptoms were the same, and the runbook did not tell the engineer where to look. We should have connected the two events earlier, and the review will look at how to make older reports easier to find.

Lessons. Small changes can cause large problems when the checks are skipped. People who are tired make fewer good decisions, so the on-call rota will be changed so that nobody is the first responder for more than 15 nights in a row. Alerts should say what to do, and not only what is wrong, and every important service needs a named owner.
