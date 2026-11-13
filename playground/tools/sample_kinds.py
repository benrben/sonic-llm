"""The ten kinds of sample document. Each function takes a Ctx (sample_words.py) and returns (text, fields, questions):
`fields` are the facts an extraction should find ({name: value}), `questions` are (question, accepted answers) about facts
stated once in the text. Every number, date, name and code in a document is different from all the others in it, so the
distractors (earlier dates, other amounts, other codes) cannot be mistaken for the answer."""

from __future__ import annotations

from sample_words import Ctx, date_forms, name_forms, number_forms

Doc = tuple[str, dict, list]


def meeting_notes(c: Ctx) -> Doc:
    club, venue, town = f"{c.pick(['Reading', 'Cycling', 'Garden', 'Chess', 'Choir', 'Allotment'])} Circle of {c.place()}", c.street(), c.place()
    chair, clerk, owner = c.person(), c.person(), c.person()
    day = c.date()
    before, nxt, due = c.before(day), c.date(after=day, within=60), c.date(after=day, within=25)
    people, quorum, budget, last_year, room = c.n(9, 24), c.n(4, 6), c.n(900, 4800), c.n(500, 899), c.n(2, 9)
    topic = c.pick(["the autumn programme", "the new storage shed", "the community open day", "the shared tools library"])
    text = f"""{club}: minutes of the committee meeting

The committee met on {c.when(day)} at the hall on {venue}, {town}. {chair} took the chair and {clerk} kept the minutes. {people} members came, which was well above the quorum of {quorum}. Apologies were received from three people who could not travel. The minutes of the meeting held on {c.when(before)} were read, agreed as a true record and signed.

1. Matters arising
The clerk reported that the letter to the council about the hall booking had been sent and acknowledged. Nothing further was needed. The chair thanked everyone who helped to tidy the hall after the last event, and asked that the cupboard in room {room} be kept clear so that the cleaners can reach the sockets.

2. Discussion of {topic}
Members spoke for about forty minutes on {topic}. Several people felt that the plan was sound but that the timetable was too tight, and a smaller group offered to look at costs before the next meeting. The chair reminded the meeting that the club spent {last_year} euros last year on similar work and that this figure had been agreed as a guide rather than a limit. After a short vote the committee approved a budget of {budget} euros for {topic}, to be spent only against receipts.

3. Actions
{owner} will collect three written quotes and send them to the clerk by {c.when(due)}. The clerk will add the quotes to the next agenda. Nobody else needs to do anything before then. It was agreed that any member may ask to see the receipts at any time, and that the treasurer will keep them in the blue folder at the hall.

4. Any other business
A member asked whether the club could offer a short beginners' session in the spring. The chair said this was a good idea and asked for a volunteer to write a proposal; no one volunteered yet, so the item stays open. There was a brief discussion of the refreshments rota, which works well, and of the noticeboard, which needs a new lock.

The meeting closed at ten past nine. The next meeting will be held on {c.when(nxt)}.
"""
    fields = {"chair": chair, "meeting_date": c.when(day), "venue_street": venue, "approved_budget_eur": str(budget), "next_meeting": c.when(nxt)}
    qa = [("Who chaired the meeting?", name_forms(chair)), ("On what date did the committee meet?", date_forms(day)),
          ("How many members came to the meeting? (not the quorum)", number_forms(people, "members")),
          (f"What budget did the committee approve for {topic}?", number_forms(budget, "euros")),
          ("Who will collect the three quotes?", name_forms(owner))]  # fmt: skip
    return text, fields, qa


def product_manual(c: Ctx) -> Doc:
    kind = c.pick(["kettle", "air purifier", "desk lamp", "slow cooker", "radio", "label printer"])
    brand, model = c.pick(["Brewmate", "Aerolite", "Quillon", "Tidewell", "Norvik", "Hearthly"]), c.code("KX", 3)
    watts, months, hold, litres, support = c.n(300, 2400), c.n(12, 36), c.n(4, 9), c.n(2, 9), c.n(100, 999)
    e1, e2, e3 = c.code("E", 2), c.code("E", 2), c.code("E", 2)
    cm, mins = c.n(10, 30), c.n(25, 50)
    text = f"""{brand} {kind}, model {model}: user guide

Thank you for choosing the {brand} {kind}. Please read this guide before you use it for the first time and keep it somewhere safe. The model number is printed on the label under the base; you will need it if you call the help line.

Setting up
Place the {kind} on a flat, dry surface at least {cm} cm from walls and other appliances. Do not put it near a sink or on a soft cloth. Remove the stickers and the foam from the box, and wipe the outside with a damp cloth. Plug it into a standard wall socket. The {kind} draws {watts} watts, so avoid sharing a single extension lead with other heavy appliances.

Everyday use
Press the round button once to start and once to stop. The light turns green while it is working and white when it is waiting. The tank holds {litres} litres and should not be filled above the marked line. Switch it off and unplug it before you clean it, and let it cool for about {mins} minutes after a long session. Never use harsh chemicals on the casing; mild soapy water is enough.

If something goes wrong
If the display shows {e1}, the filter or the inlet is blocked: switch off, unplug, clean the filter under running water and try again. The code {e2} means that the unit has overheated and is cooling down; wait until the light stops flashing. The code {e3} means the tank is empty or has been removed. Other codes are explained on the support page.

To reset the {kind} to its factory settings, hold the round button for {hold} seconds until the light blinks twice. This clears your saved preferences but does not affect the warranty.

Warranty and support
Your {kind} is covered by a warranty of {months} months from the date of purchase, as long as it has been used as described in this guide. Keep your receipt with this booklet. For help, call the support line on extension {support} during office hours, or write to the address in the box. Please have the model number ready. Repairs made by anyone other than an approved service centre are not covered.

Disposal
When the {kind} reaches the end of its life, take it to a local collection point for electrical items. Do not put it in the household bin.
"""
    fields = {"model_code": model, "warranty_months": str(months), "reset_hold_seconds": str(hold), "tank_litres": str(litres), "overheat_error_code": e2}
    qa = [("What is the model code of the product?", [model]), ("For how many months does the warranty run?", number_forms(months, "months")),
          ("How many seconds must the button be held to reset it?", number_forms(hold, "seconds")),
          ("Which error code means the unit has overheated?", [e2]), ("How many litres does the tank hold?", number_forms(litres, "litres"))]  # fmt: skip
    return text, fields, qa


def trip_plan(c: Ctx) -> Doc:
    lead, friend, dest, hub = c.person(), c.person(), c.place(), c.place()
    out = c.date()
    back = c.date(after=out, within=12)
    hotel, flight, train = f"Hotel {c.pick(['Aurelia', 'Bellevue', 'Cormorant', 'Delphine', 'Edelweiss'])}", c.code("FL"), c.code("TR")
    total, deposit, taxi, museum = c.n(900, 3800), c.n(100, 400), c.n(20, 60), c.n(8, 30)
    bag = c.n(15, 25)
    text = f"""Trip plan: a week in {dest}

Planned by {lead}, with {friend} joining for the whole trip. This page collects everything we need so that nobody has to scroll through old messages.

Dates and travel
We leave on {c.when(out)} and come back on {c.when(back)}. The outbound flight is {flight}, a morning departure from the main airport; check-in closes ninety minutes before take-off. Each of us may bring one suitcase and one small bag, and the airline allows {bag} kg for the suitcase. From {hub} we take the regional train, booking reference {train}, which runs every hour. The journey takes a little under two hours, so we should reach {dest} in the early evening.

Where we stay
We have booked {hotel}, a small family-run place near the old harbour. Breakfast is included. The room is on the second floor and has a view of the market square. The owners asked us to let them know if we will arrive after ten at night, and we have promised to send a message from the train. A deposit of {deposit} euros has been paid by {lead}; the rest is due when we check out.

Money
The whole trip is expected to cost {total} euros for the two of us, including flights, the hotel and a little spare money. A taxi from the station to the hotel costs about {taxi} euros if the buses have stopped. The museum we both want to see charges {museum} euros per person on weekdays. We agreed to keep a shared list of what each person pays and to settle up after we are home.

Things to do
- Walk along the harbour wall on the first evening and eat something simple.
- Visit the museum on the second morning, before it gets crowded.
- Take the small ferry across the bay and have lunch on the other side.
- Leave one day completely free for whatever turns out to be nice.

Before we go
Check that passports are valid, download the offline map, tell the bank we will be abroad, and put a spare charger in the hand luggage. {friend} will bring the guidebook and {lead} will print the booking pages.
"""
    fields = {"destination": dest, "departure_date": c.when(out), "hotel": hotel, "flight_code": flight, "total_cost_eur": str(total)}
    qa = [("On what date does the trip start?", date_forms(out)), ("What is the name of the hotel?", [hotel, hotel.removeprefix("Hotel ")]),
          ("What is the flight code?", [flight]), ("What is the expected total cost of the trip in euros?", number_forms(total, "euros")),
          ("Who planned the trip?", name_forms(lead))]  # fmt: skip
    return text, fields, qa


def recipe(c: Ctx) -> Doc:
    dish = c.pick(["Walnut and pear loaf", "Spiced carrot cake", "Lemon barley pudding", "Fennel and potato bake", "Oat and apple crumble", "Cheese and leek tart"])
    author, village = c.person(), c.place()
    serves, temp, bake, prep, rest, flour, butter = c.n(4, 12), c.n(160, 220), c.n(35, 65), c.n(10, 25), c.n(70, 120), c.n(200, 600), c.n(60, 190)
    eggs, sugar = c.n(2, 5), c.n(50, 140)
    text = f"""{dish}

This recipe came from {author}, who baked it every autumn in {village}. It is simple, forgiving and keeps well for a few days in a tin. It serves {serves} people.

Ingredients
- {flour} g plain flour
- {butter} g soft butter
- {sugar} g light brown sugar
- {eggs} eggs
- a pinch of salt and a teaspoon of baking powder
- a handful of chopped nuts or seeds, if you like

Method
Start by heating the oven to {temp} degrees Celsius and lining a tin with baking paper. Cream the butter and the sugar in a large bowl until the mixture is pale and light, which takes about {prep} minutes by hand. Beat in the eggs one at a time. If the mixture starts to look curdled, add a spoonful of the flour and carry on.

Fold in the flour, the salt and the baking powder with a metal spoon, using slow, wide movements so that you keep the air in. Add the nuts or seeds. Spoon the mixture into the tin and level the top with the back of the spoon.

Bake in the middle of the oven for {bake} minutes. After about half of that time, check the top; if it is browning too fast, lay a sheet of foil over it. The dish is ready when a skewer pushed into the centre comes out clean. Take it out and leave it to rest in the tin for {rest} minutes before turning it out, because it is fragile while it is hot.

Tips
Everything works better if the butter and the eggs are at room temperature. If you want a richer result, swap a spoonful of the flour for ground almonds. The leftovers can be wrapped in baking paper and kept for up to four days, and any dry end pieces make a good base for a trifle. Serve it warm with custard or cold with a cup of tea.
"""
    fields = {"dish": dish, "author": author, "servings": str(serves), "oven_temperature_celsius": str(temp), "baking_minutes": str(bake)}
    qa = [("At what oven temperature, in degrees Celsius, is the dish baked?", number_forms(temp, "degrees")),
          ("For how many minutes is it baked?", number_forms(bake, "minutes")), ("How many grams of flour does it need?", number_forms(flour, "g")),
          ("How many people does the recipe serve?", number_forms(serves, "people")), ("Who did the recipe come from?", name_forms(author))]  # fmt: skip
    return text, fields, qa


def incident_report(c: Ctx) -> Doc:
    svc, eng, lead, oncall = c.pick(["billing", "search", "login", "notification", "export"]), c.person(), c.person(), c.person()
    ident, day = c.code("INC"), c.date()
    minutes, users, retries, pages, build = c.n(12, 190), c.n(120, 9800), c.n(3, 9), c.n(2, 8), c.code("B", 4)
    cluster = c.code("CL", 3)
    cause = c.pick(["an expired certificate", "a full disk on one node", "a bad configuration value", "a stuck database migration", "a mistyped firewall rule"])
    text = f"""Incident report {ident}: the {svc} service

Summary
On {c.when(day)} the {svc} service stopped answering normal requests for {minutes} minutes. The problem was in cluster {cluster}. It was caused by {cause}, which was introduced by release {build} earlier that week. About {users} customers saw errors during the incident. No data was lost, and no customer was charged twice.

Timeline
The first alert fired at the start of the incident and was picked up within a few minutes by {oncall}, who was on call that night. {oncall} tried restarting the affected processes, which did not help, and then asked for a second pair of eyes. {eng} joined, read the logs and found the cause. A fix was prepared and reviewed by {lead}, who was the incident commander, and it was rolled out step by step. Service returned to normal and the alerts cleared.

Impact
Customers could still log in and read their existing data, but anything that needed the {svc} service failed with a generic error. Support received {pages} messages from customers; each one was answered with a short note and a link to the status page. Our automatic clients retried a request up to {retries} times before giving up, which made the load a little worse while the service was down.

What went well
The alert was clear, the on-call engineer reacted quickly, and the people involved communicated in one channel so that nobody had to guess. The rollback tool worked as designed.

What did not go well
The check that should have caught the problem before release was skipped because the change looked small. The runbook for this service had not been updated for a year, so some of its steps no longer matched the system.

Follow-up
We will add the missing check to the release pipeline, review the runbook with the team, and run a short drill so that more people know how to restart the service safely. Owners and dates are tracked in the team's planning board.
"""
    fields = {"incident_id": ident, "incident_date": c.when(day), "duration_minutes": str(minutes), "customers_affected": str(users), "incident_commander": lead}
    qa = [("What is the incident number?", [ident]), ("How many minutes did the incident last?", number_forms(minutes, "minutes")),
          ("About how many customers saw errors?", number_forms(users, "customers")), ("Who was the incident commander?", name_forms(lead)),
          ("Which cluster was affected?", [cluster])]  # fmt: skip
    return text, fields, qa


def invoice(c: Ctx) -> Doc:
    seller, buyer, contact = c.company(), c.company(), c.person()
    ident, po, issued = c.code("INV", 5), c.code("PO", 4), c.date()
    due = c.date(after=issued, within=45)
    qty1, qty2, unit1, unit2 = c.n(2, 40), c.n(2, 40), c.n(15, 90), c.n(100, 480)
    vat, ship = c.pick([7, 9, 19, 21]), c.n(11, 49)
    net = qty1 * unit1 + qty2 * unit2 + ship
    total = net + net * vat // 100
    c.used.update({net, total})
    text = f"""Invoice {ident}

From: {seller}
To: {buyer}, attention {contact}
Date of issue: {c.when(issued)}
Payment due: {c.when(due)}
Your purchase order: {po}

Description of the goods and services

Item A, box of fittings: {qty1} boxes at {unit1} euros each, which comes to {qty1 * unit1} euros.
Item B, installation kit: {qty2} kits at {unit2} euros each, which comes to {qty2 * unit2} euros.
Delivery and packing: {ship} euros.

Subtotal before tax: {net} euros.
Tax at {vat} percent: {net * vat // 100} euros.
Total amount due: {total} euros.

Payment terms
Please pay the total amount by bank transfer by the due date and quote the invoice number in the reference field. The bank details are on the last page of our price list. A late payment fee may be added to any invoice that remains unpaid after the due date, and we will send a friendly reminder before that happens.

Notes
All goods were delivered to your warehouse in one shipment and signed for by your receiving clerk. If anything is missing or damaged, please tell us within ten working days and we will arrange a replacement or a credit note. Prices are fixed for the current order. For questions about this invoice, contact our accounts team and mention the purchase order above, which helps us find your order quickly.

Thank you for your business. We look forward to working with {buyer} again.
"""
    fields = {"invoice_number": ident, "customer": buyer, "due_date": c.when(due), "purchase_order": po, "total_eur": str(total)}
    qa = [("What is the invoice number?", [ident]), ("What is the total amount due in euros?", number_forms(total, "euros")),
          ("By what date must the invoice be paid?", date_forms(due)), ("Which purchase order does the invoice refer to?", [po]),
          ("Which company is billed?", [buyer, buyer.split()[0]])]  # fmt: skip
    return text, fields, qa


def lease_summary(c: Ctx) -> Doc:
    tenant, landlord, street, town = c.person(), c.person(), c.street(), c.place()
    number, flat = c.n(3, 98), c.n(1, 9)
    start = c.date()
    rent, deposit, term, notice, fee = c.n(650, 2300), c.n(700, 2600), c.pick([6, 12, 18, 24]), c.n(30, 90), c.n(20, 80)
    c.used.add(term)
    heating = c.n(40, 120)
    text = f"""Summary of the residential lease at {number} {street}, flat {flat}, {town}

This summary explains the main points of the lease in plain language. If this summary and the signed lease disagree, the signed lease is the one that counts.

The people involved
The landlord is {landlord}. The tenant is {tenant}. The landlord lives in the same town and can be reached at the number written in the lease.

Money
The monthly rent is {rent} euros, paid in advance on the first working day of each month by standing order. A security deposit of {deposit} euros was paid before the keys were handed over. It will be returned within a month after the tenant leaves, less any fair deduction for damage beyond normal wear. In addition to the rent, the tenant pays about {heating} euros a month toward heating, which is reviewed once a year against the real bills. If the rent is more than ten days late, the landlord may charge a fee of {fee} euros.

Length of the lease
The lease starts on {c.when(start)} and runs for {term} months. After that period it continues from month to month unless either side ends it. Either side can end the lease by giving notice in writing; the notice period is {notice} days. The tenant may not end the lease during the first three months except by agreement.

Rules of the house
The flat is let for living in and not for business. A pet may be kept only with the landlord's written permission. Smoking is not allowed inside. The tenant must keep the stairs and the shared bicycle room tidy on the weeks set out in the cleaning rota, and must not make loud noise after ten at night. Small repairs, such as changing light bulbs, are the tenant's job; larger repairs should be reported to the landlord at once.

Inspection
The landlord may inspect the flat with at least two days' notice and at a reasonable time. A photo list of the condition of each room was signed on the day of the handover, and both people kept a copy.
"""
    fields = {"tenant": tenant, "landlord": landlord, "monthly_rent_eur": str(rent), "start_date": c.when(start), "notice_days": str(notice)}
    qa = [("What is the monthly rent in euros?", number_forms(rent, "euros")), ("How large is the security deposit in euros?", number_forms(deposit, "euros")),
          ("On what date does the lease start?", date_forms(start)), ("How many days of notice are needed to end the lease?", number_forms(notice, "days")),
          ("Who is the tenant?", name_forms(tenant))]  # fmt: skip
    return text, fields, qa


def lab_report(c: Ctx) -> Doc:
    tech, sup = c.person(), c.person()
    day, batch, sample = c.date(), c.code("BT", 4), c.code("SM", 3)
    inst = f"{c.pick(['Veltro', 'Orbis', 'Nanex', 'Kalder'])} {c.n(100, 900)}"
    n, temp, ph10, conc10 = c.n(6, 24), c.n(18, 30), c.n(66, 82), c.n(120, 980)
    ph, conc = f"{ph10 // 10}.{ph10 % 10}", f"{conc10 // 10}.{conc10 % 10}"
    ref, blank = c.n(2, 5), f"{c.n(1, 9)}.{c.n(1, 9)}"
    text = f"""Laboratory report, batch {batch}

Purpose
This report describes the routine check of water samples from the east pond, taken to confirm that the acidity and the dissolved mineral content stay within the limits agreed with the site manager. The work was carried out by {tech} and reviewed by {sup}.

Samples
{n} samples were collected on {c.when(day)} in clean, labelled bottles and brought to the laboratory within two hours. The reference sample, labelled {sample}, was kept at the same temperature as the others and treated in exactly the same way. The room temperature during the measurements was {temp} degrees Celsius.

Method
Each sample was shaken gently, left to settle for five minutes and then measured with the {inst}. The instrument was calibrated before the first sample using two standard solutions, and checked again after the last one; the drift was below the tolerance. Every sample was measured {ref} times and the average was written down. A blank sample of distilled water gave a reading of {blank}, which was subtracted where needed.

Results
The average acidity of the samples was pH {ph}. The mineral content of the reference sample {sample} was {conc} mg/L. The other samples were close to it, and none of them stood out from the group. There were no signs of contamination, and no bottle was cloudy or had a visible film on the surface.

Discussion
The acidity is typical for this kind of pond and gives no cause for concern. The mineral content is a little lower than in the spring, which is expected after a wet period. We recommend repeating the check next quarter and keeping the same instrument for the measurement, so that the numbers can be compared with confidence.

Notes
Bottles and equipment were washed and rinsed with distilled water. Waste solutions were collected in the marked container. The raw numbers are stored in the laboratory notebook, and a copy of this report was handed to the site manager.
"""
    fields = {"batch": batch, "date": c.when(day), "technician": tech, "instrument": inst, "mean_ph": ph}
    qa = [("What is the batch code?", [batch]), ("What was the average pH of the samples?", [ph, f"pH {ph}"]),
          (f"What was the mineral content of reference sample {sample} in mg/L?", [f"{conc} mg/L", conc]), ("Which instrument was used for the measurements?", [inst]),
          ("Who carried out the work?", name_forms(tech))]  # fmt: skip
    return text, fields, qa


def club_newsletter(c: Ctx) -> Doc:
    club, pres, editor, town, hall = f"{c.place()} {c.pick(['Rowing', 'Walking', 'Quilting', 'Birdwatching', 'Swimming'])} Club", c.person(), c.person(), c.place(), c.street()
    month = c.pick(["March", "May", "June", "September", "October", "November"])
    event = c.date(within=90)
    deadline = c.before(event, 5, 25)
    fee, members, prize, kids, volunteers = c.n(8, 45), c.n(60, 340), c.n(40, 250), c.n(5, 20), c.n(4, 14)
    text = f"""{club}: the {month} newsletter

A word from the president
Dear members, welcome to another newsletter. {pres}, our president, writes: "The club has grown to {members} members this year, and I want to thank everyone who has helped with the boats, the paths and the kettle. A club is only as good as the people who turn up on a wet morning."

The big event
Our annual open day will take place on {c.when(event)} at the {hall} grounds in {town}. There will be short guided sessions for beginners, a tea tent, a book swap and a small raffle. The first prize in the raffle is a voucher worth {prize} euros, donated by a local shop. Entry for members is free; guests pay {fee} euros, which covers the tea and a place in one of the sessions.

To take part in a session, please sign up by {c.when(deadline)} using the list on the noticeboard or by sending a message to the club address. Places for the children's session are limited to {kids}, and they are given out in the order that names arrive. We still need {volunteers} volunteers to help with parking, tea and tidying up, so if you have an hour to spare, please say so.

Club news
The winter repairs are finished and the shed has a new roof. The committee thanks the members who donated their weekends. The photo wall in the entrance has been refreshed with pictures from last year; if you spot one of yourself and would like a copy, just ask. New members are always welcome, and an introductory evening is held every first Thursday.

Reminders
Please return borrowed equipment cleaned and dry. Park on the side of the road marked for visitors so that the lane stays clear for deliveries. The kettle and the mugs are for everybody, and so is the washing up.

This newsletter was put together by {editor}. If you have an item for next time, send it in before the end of the month.
"""
    fields = {"club": club, "president": pres, "event_date": c.when(event), "guest_fee_eur": str(fee), "signup_deadline": c.when(deadline)}
    qa = [("On what date is the open day?", date_forms(event)), ("How much do guests pay in euros?", number_forms(fee, "euros")),
          ("Who is the president of the club?", name_forms(pres)), ("By what date must people sign up for a session?", date_forms(deadline)),
          ("How many members does the club have?", number_forms(members, "members"))]  # fmt: skip
    return text, fields, qa


def project_status(c: Ctx) -> Doc:
    name, pm, owner, dev = f"Project {c.pick(['Lantern', 'Harbour', 'Meadow', 'Compass', 'Anvil', 'Orchard'])}", c.person(), c.person(), c.person()
    week = c.date()
    miles = c.date(after=week, within=90)
    spent, total, bugs, closed, people = c.n(20, 90), c.n(100, 400), c.n(3, 40), c.n(41, 120), c.n(5, 18)
    spent_k, total_k = spent * 1000, total * 1000
    c.used.update({spent_k, total_k})
    blocker = c.pick(["the test server", "the data export", "the legal review", "the supplier contract"])
    text = f"""{name}: weekly status

Week ending {c.when(week)}. Written by {pm}, project manager, for the steering group.

Overall
The project is on track but needs attention in one area. {people} people worked on it this week. The team finished the planned work on the first screen and started on the second one. The mood in the team is good, and the daily check-in is working well.

Money
So far {spent_k} euros of the total budget of {total_k} euros have been spent. This is in line with the plan for this point in time. No extra spending has been approved. The finance team will send the next report at the end of the month.

Milestones
The next milestone, the first version for real users, is due on {c.when(miles)}. The team believes this is still realistic. The milestone after it is not yet scheduled, and will be discussed once the first version is in the hands of users.

Quality
{closed} bugs have been closed since the project began. {bugs} bugs are still open, and none of them stops people from using the product. The most serious open bugs are a layout problem on small screens and a slow search. {dev} is working on both and expects to finish them next week.

Risks and blockers
One blocker is open: we are waiting for {blocker}. {owner} is the owner of this blocker and has been chasing it for several days. If it is not solved soon, the milestone may move by a week or two. The steering group is asked to help by making a quick decision when the answer arrives.

Next week
Finish the second screen, fix the layout bug, write the user guide and run a first test with three friendly users. {pm} will send a short note on Friday to say how it went.
"""
    fields = {"project": name, "project_manager": pm, "next_milestone_date": c.when(miles), "open_bugs": str(bugs), "blocker_owner": owner}
    qa = [("Who is the project manager?", name_forms(pm)), ("On what date is the next milestone due?", date_forms(miles)),
          ("How many bugs are still open?", number_forms(bugs, "bugs")), ("Who owns the blocker?", name_forms(owner)),
          ("How many euros of the budget have been spent so far?", number_forms(spent_k, "euros"))]  # fmt: skip
    return text, fields, qa


KINDS = {"meeting notes": meeting_notes, "product manual": product_manual, "trip plan": trip_plan, "recipe": recipe,
         "incident report": incident_report, "invoice": invoice, "lease summary": lease_summary, "lab report": lab_report,
         "club newsletter": club_newsletter, "project status": project_status}  # fmt: skip
