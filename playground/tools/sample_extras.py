"""Extra paragraphs for the sample documents: ordinary text with distractor numbers, dates and names, so that a document is
longer and an answer cannot be guessed from the digits alone. Placeholders are filled from the document's Ctx and are all new:
{n1}..{n3} small numbers, {big} a larger one, {deg} a temperature, {hour} an hour, {age}, {cm}, {date}, {person}, {place}, {company}, {code}."""

from __future__ import annotations

from sample_words import Ctx


class _Fill(dict):
    """format_map source: every placeholder gets a fresh value from the document's Ctx, kept for repeats of the same name."""

    def __init__(self, c: Ctx):
        super().__init__()
        self.c = c

    def __missing__(self, key: str) -> str:
        c = self.c
        makers = {"n": lambda: str(c.n(3, 99)), "big": lambda: str(c.n(120, 980)), "deg": lambda: str(c.n(14, 31)), "hour": lambda: str(c.n(6, 11)), "age": lambda: str(c.n(6, 14)), "cm": lambda: str(c.n(18, 30)),
                  "date": lambda: c.when(c.date()), "person": c.person,
                  "place": c.place, "company": c.company, "code": lambda: c.code("RF", 3)}
        self[key] = makers[key.rstrip("0123456789")]()
        return self[key]


EXTRAS = {
    "meeting notes": [
        "Treasurer's note. The treasurer reported {big} euros in the current account at the end of last month and a further {n1} euros in the cash box. Membership fees have been collected from {n2} households so far, and a reminder will go to the rest. Nothing unusual was spent in the quarter. The treasurer asked members to hand in receipts within a fortnight, because the accounts are checked by {person} each spring.",
        "The hall. The caretaker, {person}, asked the committee to remember that the hall is not free on Tuesdays and that it holds {n1} chairs and {n2} tables. A leaking tap in the kitchen has been mended. The heating will be switched on from the second week of October, and the thermostat must not be touched by anyone except the caretaker, whatever the temperature in the room.",
        "Correspondence. A letter arrived from {company} offering a discount on printing. The clerk will keep it on file and the committee will look at it when the next newsletter is due. A thank-you card came from the family of a former member. A short email from {person}, who lives in {place}, asked whether visitors from other clubs may join our outings; the chair will reply that they are welcome.",
        "Welfare. The committee agreed to send flowers to a member who is recovering from an operation and to visit on the following weekend. About {n1} cards were signed at the last event and will be posted. The chair asked everyone to keep an eye on older members who may find the winter hard, and to pass on any worries quietly to the clerk.",
        "Social events. The summer outing to {place} went well, with {n1} people on the coach and good weather all day. A short quiz evening is planned for the dark months. {person} suggested a joint evening with a neighbouring group, and members liked the idea, although nothing was fixed. Cake, tea and a small raffle are expected, as always.",
    ],
    "product manual": [
        "Safety. Use the product only indoors and only for the purpose described in this guide. Keep it out of the reach of children under {age} years. Do not let the cord hang over the edge of a table. If the cord or the plug is damaged, stop using the product and have it replaced by the service centre. Never open the casing yourself: there are no parts inside that you can repair, and it may void the warranty.",
        "Cleaning and care. Clean the filter every {n1} weeks, or sooner if you notice a change in performance. Do not put any part in the dishwasher unless it is marked as safe for it. Dry all parts completely before you put them back. If you will not use the product for a long time, store it in its box in a cool, dry place, with the cord wrapped loosely and not pulled tight.",
        "Accessories. A spare filter costs {n1} euros and can be ordered with the part code {code} from the shop or from any approved dealer. A carrying bag and a second power cord are also available. Only use accessories made for this product; others may not fit well and can cause damage. The packaging contains {n2} small screws for the wall bracket and a short instruction sheet.",
        "Frequently asked questions. Why does the product make a low humming sound? This is normal and comes from the motor. Why does it smell a little when new? A new product can give off a mild smell for the first {n1} hours of use, which disappears with ventilation. Can I use it in a bathroom? We do not recommend this, because moisture and electricity do not mix well.",
        "Environment. The packaging is made from recycled cardboard and can be recycled again. The product uses little energy when it is waiting, about {n1} percent of its working power. We design our products to last, and replacement parts are kept for several years after a model is sold. Please do not throw batteries or electronics into the household waste.",
    ],
    "trip plan": [
        "Packing. We will take light clothes that can be layered, a rain jacket, comfortable walking shoes and one nicer outfit for the evening. The weather forecast promises around {deg} degrees in the afternoon and cooler nights, with a chance of rain on two days. Sun cream, a small first aid kit, a reusable water bottle and an adapter for the sockets also go in the bag.",
        "Eating out. A colleague recommended a small restaurant called {company} near the market, which has a short menu and fair prices. We expect to spend about {n1} euros per person on a good dinner and much less on lunch. Cafes in {place} often close early on Sundays, so we will buy bread and fruit the day before if we want a picnic.",
        "Getting around. The old town is small and easy to walk. A day ticket for local buses costs {n1} euros, and the ticket machine at the station takes cards. Bicycles can be rented near the harbour, but the hills are steeper than they look. {person}, who visited last year, warns that the cobbled streets are hard on suitcase wheels.",
        "Emergencies. The local number for the ambulance is {big}, and the nearest pharmacy is two streets from the hotel. We have both taken out travel insurance, and a copy of the policy is saved in the shared folder. The bank card helpline is also written there. If one of us gets lost, we will meet at the fountain on the market square.",
        "Wish list. If there is time we would like to see the lighthouse, walk the hill path to the ruined castle, and find a shop that sells local ceramics. {person} recommended the sunset from the old pier, and a small gallery that is open until {hour} in the evening. We will decide day by day and not plan every hour.",
    ],
    "recipe": [
        "A little history. The recipe has been in the family for many years. It was written on the back of an envelope, in pencil, and then copied into a notebook that now has {n1} pages of recipes in all. Over the years the sugar has been reduced and a little spice added. Friends always ask for the recipe, and it has been passed on to at least {n2} households.",
        "Storing and freezing. Wrap the cooled dish well in baking paper and then in foil. It keeps for several days in a cool place, and it can be frozen for up to {n1} weeks. Thaw it at room temperature for a few hours and warm it for a few minutes in a low oven if you want it to taste freshly baked.",
        "Variations. Replace the nuts with dried fruit, or add the grated zest of one lemon for a fresher taste. For a vegan version use a plant-based spread and the same weight of mashed banana in place of the eggs. Some bakers brush the top with {n1} g of warm honey as soon as it comes out of the oven, which gives it a shine.",
        "Equipment. A medium tin about {cm} cm wide works best. A hand whisk is enough, but an electric mixer saves time and effort. Use an oven thermometer if your oven runs hot or cold, because small differences in temperature can change the result. Always let the tin cool a little before you wash it.",
        "A note on ingredients. Use the best butter you can find, since it is the main flavour. Eggs of any size are fine, but do not use cold ones straight from the fridge. Flour from {place} mills is a little coarser and takes a little more liquid. If the mixture is too stiff, add a spoonful of milk; if it is too soft, a spoonful of flour.",
    ],
    "incident report": [
        "Metrics. The service normally answers in about {n1} milliseconds and fails fewer than {n2} requests in a thousand. During the incident the failure rate rose sharply and the average answer time went up more than tenfold. Within {n3} minutes of the fix being applied, both numbers returned to normal values. Graphs of the period are attached to the ticket.",
        "Communication. The status page was updated three times, and each update was reviewed by {person} before it was published. An internal message went to the wider team, and a short note was sent to customers who had contacted support. We believe the updates were honest and clear, although one of them used a technical term that customers did not know.",
        "Earlier incidents. A similar problem occurred some months ago and was recorded as ticket {code}. At that time the cause was different, but the symptoms were the same, and the runbook did not tell the engineer where to look. We should have connected the two events earlier, and the review will look at how to make older reports easier to find.",
        "Lessons. Small changes can cause large problems when the checks are skipped. People who are tired make fewer good decisions, so the on-call rota will be changed so that nobody is the first responder for more than {n1} nights in a row. Alerts should say what to do, and not only what is wrong, and every important service needs a named owner.",
        "Costs. The incident used about {n1} hours of engineering time in total, spread over several people. We offered no refunds, since nobody was charged for the time of the outage. The cost of the extra servers we started during the recovery was small, and they were shut down again the following day. The review meeting will take one hour.",
    ],
    "invoice": [
        "Delivery details. The goods left our warehouse in {place} on a pallet weighing {big} kg, packed in {n1} cartons. The carrier's reference is {code}. The delivery note was signed on arrival, and the photo of the signed note can be sent on request. The carrier delivers between eight and five on working days and does not call ahead unless the address is hard to find.",
        "Earlier orders. Your previous order was delivered about {n1} weeks ago and was paid on time. We would like to thank you for the prompt payment. If you plan to order again soon, please tell us in advance so that we can keep enough stock. Standing orders can be set up with a monthly delivery and a single monthly invoice, which many customers find easier.",
        "About us. We have been supplying small businesses for more than {n1} years. Our warehouse is open to visitors by appointment, and our staff are happy to help with advice on which products suit which jobs. We aim to answer every question within one working day, and our accounts team is available by telephone on weekday mornings.",
        "Returns. Unused goods in their original packing can be returned within {n1} days of delivery. Please ask for a returns number first and attach it to the parcel. The cost of return postage is paid by the buyer, unless the goods were wrong or damaged. A credit note will be issued within a week of the returned goods arriving.",
        "Data protection. We keep your name, address and order history only as long as necessary to run our business and to meet legal duties. We do not sell your details. You may ask to see, correct or delete them at any time by writing to {person}, who looks after data protection for us. Invoices must be kept for several years under tax law.",
    ],
    "lease summary": [
        "Utilities and services. Water and the shared electricity in the hallway are included in the heating charge. The tenant arranges and pays for their own electricity and internet, and must give the meter readings on the day they move in; the starting reading on the electricity meter was {big}. Rubbish collection is on {n1} days of the month, and bins must be put out the evening before.",
        "Parking and storage. There is room for one bicycle per person in the shared room on the ground floor, and a small cellar compartment, number {n1}, belongs to the flat. No parking space is included, but residents may apply to the town for a permit. The landlord asks that nothing is stored on the stairs or in the corridor because of the fire rules.",
        "Insurance. The landlord insures the building and the shared parts. The tenant is advised to take out a contents insurance policy and to include cover for accidental damage and water. A copy of the policy should be handed to the landlord on request. A neighbour, {person}, has agreed to keep a spare key for emergencies, and the landlord knows about this arrangement.",
        "Changes and repairs. The tenant may paint the walls in light colours if the flat is returned in white at the end. Drilling holes for shelves is allowed, but large nails and sticky hooks on tiles are not. The flat has {n1} windows, all of which were checked by the landlord before the tenant moved in. Broken glass from normal use is the tenant's cost.",
        "Ending the lease. At the end, the tenant returns all keys, cleans the flat and sets up a time for the final inspection with the landlord. Any difference between the first photo list and the final state will be discussed in a friendly way. The landlord keeps a list of about {n1} recommended cleaners, and the tenant may use any of them or do the work personally.",
    ],
    "lab report": [
        "Safety. All the people in the room wore gloves, safety glasses and lab coats. The room was ventilated during the work, and the instrument was placed away from the sink. Waste liquids were poured into the labelled container and handed over to {person}, who is in charge of waste. No spills occurred, and no first aid was needed.",
        "Quality control. A duplicate of every {n1}th sample was measured again, and the results agreed within the allowed range. The balance was checked with a standard weight of {n2} g before use. The distilled water used for rinsing comes from the laboratory tap, which is tested every month. Records of the checks are kept in the folder for this year.",
        "Equipment and maintenance. The instrument was last serviced about {n1} months ago and the probe was replaced at that time. The probe is stored in its special liquid between uses and never left to dry. If the readings begin to wander, the probe should be cleaned first and replaced second. The next service is planned for the autumn.",
        "Background. The east pond was dug many years ago and is fed by rain and by a small stream. It is used for watering the garden and by wildlife, and it has been sampled by the site team every quarter for the last {n1} years. The notebook from the first year shows slightly different readings, but the method was not the same.",
        "Comments from the reviewer. The report is clear and the numbers are easy to follow. The reviewer asked to add the temperature of the samples at the time of collection, and a short sentence about the weather on that day, because rain can change the readings. {person} will keep a copy of the raw data for at least {n1} years.",
    ],
    "club newsletter": [
        "Dates for the diary. The monthly walk starts at ten o'clock on the first Saturday and covers about {n1} kilometres. The committee meets on the second Wednesday of the month. A repair weekend is planned for {date}, and a social evening for {date2}. All the dates will be repeated on the noticeboard and on the club's web page, so please check them before you travel.",
        "Lost and found. Over the summer the following items were left at the clubhouse: a blue water bottle, one glove, a pair of reading glasses and a folded umbrella. They are in a box by the door. Anything that is not collected within {n1} weeks will go to the charity shop in {place}. Please look in the box, even if you think you have lost nothing.",
        "Thank you. Special thanks to {person} for painting the fence, to {person2} for mending the gate and to everyone who brought cakes. The bake sale in the spring raised {big} euros, which will go toward new equipment. The club also received a kind letter from {company}, which has offered to give us a discount on supplies.",
        "Member profile. This month we meet {person}, who joined the club {n1} years ago and has not missed a single season since. Asked what keeps him coming back, he said that it is the people, the fresh air and the tea at the end. He has a favourite spot near the old tree, and he has promised to show new members around.",
        "Tips of the month. Bring a spare pair of dry socks, since the paths can be wet. Check the weather before you leave and tell somebody where you are going. Water is available at the clubhouse, but not along the way. If you are bringing a child, please let us know in advance so that we have enough helpers and equipment for the {n1} youngest members.",
    ],
    "project status": [
        "Team notes. A new colleague, {person}, joined the team on Monday and is already settling in. The team will move to a larger room next month. A short workshop on testing was held on Wednesday and {n1} people attended. The team agreed to keep the daily check-in to fifteen minutes and to write down the decisions in the shared notes.",
        "Dependencies. The project depends on two other teams. The first, led by {person}, provides the data and has promised an update in a few days. The second provides the login service, which has been stable for {n1} weeks. If either of them changes its plan, we will tell the steering group at once, because a delay could affect the date of the first release.",
        "Decisions made this week. The team decided to keep the old colour scheme for the first version and to postpone the dark mode to a later release. It also decided to use the existing hosting service, which saves about {n1} days of work. The decisions were taken by consensus, and nobody asked for a vote. They are recorded in the decision log under {code}.",
        "Feedback. Early users told us that the first screen is clear, but the buttons on the second are too small. We will fix this in the next sprint. One user, who works at {company}, asked for an export to a spreadsheet, and the team will consider it after the first version. In total {n1} people have given feedback so far, and most comments were positive.",
        "Communication. The weekly status is sent to the steering group every Friday. The team keeps a public board where everybody can see what is in progress. If something blocks the work for more than a day, it is raised in the daily check-in and, if needed, with the steering group. Next month we will hold a short review to see how the project is going overall.",
    ],
}


def extras(c: Ctx, kind: str, k: int = 4) -> str:
    """`k` extra paragraphs for a document of `kind`, picked and filled from the document's own random source."""
    pool = list(EXTRAS[kind])
    c.r.shuffle(pool)
    fill = _Fill(c)
    return "\n\n".join(p.format_map(fill) for p in pool[:k])
