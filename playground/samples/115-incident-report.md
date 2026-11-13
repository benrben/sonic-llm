Incident report INC-4940: the notification service

Summary
On February 5, 2027 the notification service stopped answering normal requests for 60 minutes. The problem was in cluster CL-306. It was caused by an expired certificate, which was introduced by release B-8493 earlier that week. About 5497 customers saw errors during the incident. No data was lost, and no customer was charged twice.

Timeline
The first alert fired at the start of the incident and was picked up within a few minutes by Jonas Fontaine, who was on call that night. Jonas Fontaine tried restarting the affected processes, which did not help, and then asked for a second pair of eyes. Pavel Holloway joined, read the logs and found the cause. A fix was prepared and reviewed by Hugo Iversen, who was the incident commander, and it was rolled out step by step. Service returned to normal and the alerts cleared.

Impact
Customers could still log in and read their existing data, but anything that needed the notification service failed with a generic error. Support received 7 messages from customers; each one was answered with a short note and a link to the status page. Our automatic clients retried a request up to 6 times before giving up, which made the load a little worse while the service was down.

What went well
The alert was clear, the on-call engineer reacted quickly, and the people involved communicated in one channel so that nobody had to guess. The rollback tool worked as designed.

What did not go well
The check that should have caught the problem before release was skipped because the change looked small. The runbook for this service had not been updated for a year, so some of its steps no longer matched the system.

Costs. The incident used about 14 hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.

Lessons. Small changes can cause large problems when the checks are skipped. People who are tired make fewer good decisions, so the on-call rota will be changed so that nobody is the first responder for more than 14 nights in a row. Alerts should say what to do, and not only what is wrong, and every important service needs a named owner.

Earlier incidents. A similar problem occurred some months ago and was recorded as ticket RF-927. At that time the cause was different, but the symptoms were the same, and the runbook did not tell the engineer where to look. We should have connected the two events earlier, and the review will look at how to make older reports easier to find.

Metrics. The service normally answers in about 14 milliseconds and fails fewer than 97 requests in a thousand. During the incident the failure rate rose sharply and the average answer time went up more than tenfold. Within 9 minutes of the fix being applied, both numbers returned to normal values. Graphs of the period are attached to the ticket.

Follow-up
We will add the missing check to the release pipeline, review the runbook with the team, and run a short drill so that more people know how to restart the service safely. Owners and dates are tracked in the team's planning board.
