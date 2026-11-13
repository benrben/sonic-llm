Incident report INC-1976: the login service

Summary
On March 29, 2027 the login service stopped answering normal requests for 100 minutes. The problem was in cluster CL-216. It was caused by a stuck database migration, which was introduced by release B-1651 earlier that week. About 5378 customers saw errors during the incident. No data was lost, and no customer was charged twice.

Timeline
The first alert fired at the start of the incident and was picked up within a few minutes by Clara Eriksen, who was on call that night. Clara Eriksen tried restarting the affected processes, which did not help, and then asked for a second pair of eyes. Olga Novak joined, read the logs and found the cause. A fix was prepared and reviewed by Oskar Olsen, who was the incident commander, and it was rolled out step by step. Service returned to normal and the alerts cleared.

Impact
Customers could still log in and read their existing data, but anything that needed the login service failed with a generic error. Support received 7 messages from customers; each one was answered with a short note and a link to the status page. Our automatic clients retried a request up to 4 times before giving up, which made the load a little worse while the service was down.

What went well
The alert was clear, the on-call engineer reacted quickly, and the people involved communicated in one channel so that nobody had to guess. The rollback tool worked as designed.

What did not go well
The check that should have caught the problem before release was skipped because the change looked small. The runbook for this service had not been updated for a year, so some of its steps no longer matched the system.

Communication. The status page was updated three times, and each update was reviewed by Celia Kessler before it was published. An internal message went to the wider team, and a short note was sent to customers who had contacted support. We believe the updates were honest and clear, although one of them used a technical term that customers did not know.

Lessons. Small changes can cause large problems when the checks are skipped. People who are tired make fewer good decisions, so the on-call rota will be changed so that nobody is the first responder for more than 23 nights in a row. Alerts should say what to do, and not only what is wrong, and every important service needs a named owner.

Costs. The incident used about 23 hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.

Metrics. The service normally answers in about 23 milliseconds and fails fewer than 72 requests in a thousand. During the incident the failure rate rose sharply and the average answer time went up more than tenfold. Within 90 minutes of the fix being applied, both numbers returned to normal values. Graphs of the period are attached to the ticket.

Follow-up
We will add the missing check to the release pipeline, review the runbook with the team, and run a short drill so that more people know how to restart the service safely. Owners and dates are tracked in the team's planning board.
