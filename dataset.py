"""
A small, hand-labeled dataset of spam/ham SMS-style messages, embedded
directly so the project has no external download dependency. Realistic
in style (modeled on the well-known SMS Spam Collection patterns) but
written from scratch for this repo.
"""

DATA: list[tuple[str, str]] = [
    ("Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!", "spam"),
    ("URGENT: Your account has been suspended. Verify your details immediately to avoid closure.", "spam"),
    ("FREE entry into our $5000 prize draw, just text WIN to 80086 now!", "spam"),
    ("You have been selected for a free cruise to the Bahamas! Call now to claim.", "spam"),
    ("Get cheap loans approved in minutes, no credit check required, apply now!", "spam"),
    ("WINNER!! As a valued customer you have been selected to receive a free iPhone.", "spam"),
    ("Claim your free voucher now, limited time offer, click the link below!", "spam"),
    ("Hot singles in your area want to chat with you tonight, click here!", "spam"),
    ("Your loan has been pre-approved for $10000, reply YES to proceed instantly.", "spam"),
    ("Act now! Your prize of $2000 expires in 24 hours, claim immediately.", "spam"),
    ("Earn $500 a day working from home, no experience needed, sign up free.", "spam"),
    ("Final notice: your subscription payment failed, update your card details now.", "spam"),
    ("Text STOP to unsubscribe or continue to receive exclusive deals and discounts daily.", "spam"),
    ("You've been chosen for a free trial of our weight loss pills, order today.", "spam"),
    ("Limited offer: buy one get one free on all electronics, shop now before it ends.", "spam"),
    ("Your package could not be delivered, click here to reschedule and pay a small fee.", "spam"),
    ("Congratulations, you have qualified for a cash reward, call this number now.", "spam"),
    ("Get rich quick with this one simple trick, millions of people are doing it.", "spam"),
    ("Your bank account needs verification, please confirm your PIN and card number.", "spam"),
    ("Free ringtones and wallpapers, just download our app and register your number.", "spam"),
    ("Hey, are we still meeting for lunch tomorrow at noon?", "ham"),
    ("Can you pick up some milk and eggs on your way home tonight?", "ham"),
    ("Thanks for helping me move this weekend, I really appreciate it.", "ham"),
    ("The meeting got pushed to 3pm, see you in conference room B.", "ham"),
    ("Happy birthday! Hope you have an amazing day, let's celebrate this weekend.", "ham"),
    ("I finished the report, can you review it before I send it to the client?", "ham"),
    ("Running a bit late, traffic is bad, should be there in 15 minutes.", "ham"),
    ("Don't forget we have the dentist appointment on Thursday morning.", "ham"),
    ("Loved the movie last night, we should watch the sequel next week.", "ham"),
    ("Mom said dinner is at 7, don't be late this time.", "ham"),
    ("Can you send me the notes from today's lecture, I missed the second half?", "ham"),
    ("The package arrived today, thanks for ordering it for me.", "ham"),
    ("Let's go for a run tomorrow morning before it gets too hot.", "ham"),
    ("I'll be working from home today, call me if anything urgent comes up.", "ham"),
    ("Great game last night, can't believe that final shot in overtime.", "ham"),
    ("Just landed, will text you once I'm out of the airport.", "ham"),
    ("Reminder: rent is due on the first, let me know if you need an extension.", "ham"),
    ("Can we reschedule our call to Friday instead of Wednesday?", "ham"),
    ("Thanks for the recipe, the pasta turned out great, everyone loved it.", "ham"),
    ("I left my charger at your place, can I swing by later to grab it?", "ham"),
]


def load() -> tuple[list[str], list[str]]:
    texts = [t for t, _ in DATA]
    labels = [l for _, l in DATA]
    return texts, labels
