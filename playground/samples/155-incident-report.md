Incident report INC-5863: the login service

Summary
On February 3, 2027 the login service stopped answering normal requests for 163 minutes. The problem was in cluster CL-779. It was caused by an expired certificate, which was introduced by release B-4845 earlier that week. About 2393 customers saw errors during the incident. No data was lost, and no customer was charged twice.

Timeline
The first alert fired at the start of the incident and was picked up within a few minutes by Sven Wexler, who was on call that night. Sven Wexler tried restarting the affected processes, which did not help, and then asked for a second pair of eyes. Rafael Ulbricht joined, read the logs and found the cause. A fix was prepared and reviewed by Selma Vasquez, who was the incident commander, and it was rolled out step by step. Service returned to normal and the alerts cleared.

Impact
Customers could still log in and read their existing data, but anything that needed the login service failed with a generic error. Support received 2 messages from customers; each one was answered with a short note and a link to the status page. Our automatic clients retried a request up to 9 times before giving up, which made the load a little worse while the service was down.

What went well
The alert was clear, the on-call engineer reacted quickly, and the people involved communicated in one channel so that nobody had to guess. The rollback tool worked as designed.

What did not go well
The check that should have caught the problem before release was skipped because the change looked small. The runbook for this service had not been updated for a year, so some of its steps no longer matched the system.

Earlier incidents. A similar problem occurred some months ago and was recorded as ticket RF-352. At that time the cause was different, but the symptoms were the same, and the runbook did not tell the engineer where to look. We should have connected the two events earlier, and the review will look at how to make older reports easier to find.

Lessons. Small changes can cause large problems when the checks are skipped. People who are tired make fewer good decisions, so the on-call rota will be changed so that nobody is the first responder for more than 62 nights in a row. Alerts should say what to do, and not only what is wrong, and every important service needs a named owner.

Costs. The incident used about 62 hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.

Communication. The status page was updated three times, and each update was reviewed by Pavel Jankowski before it was published. An internal message went to the wider team, and a short note was sent to customers who had contacted support. We believe the updates were honest and clear, although one of them used a technical term that customers did not know.

Follow-up
We will add the missing check to the release pipeline, review the runbook with the team, and run a short drill so that more people know how to restart the service safely. Owners and dates are tracked in the team's planning board.
