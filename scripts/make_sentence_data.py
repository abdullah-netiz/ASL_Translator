"""Generate clean ASL-gloss -> English training pairs for the composer from a small English grammar over the
recognizer's vocabulary. Glosses follow common ASL conventions: time word first, WH-word last, NOT before the
verb, no articles/copula/auxiliaries; "YOU ..." without a WH-word is a yes/no question.
Any pair whose gloss appears in the hand-written test set (data/sentences/eval_gloss_en.tsv) is dropped.
python scripts/make_sentence_data.py  -> data/sentences/{train,val}.tsv"""
import random, csv, itertools
from pathlib import Path

R = random.Random(0)
OUT = Path("data/sentences")

# ---------------------------------------------------------------- lexicon
FAMILY = "mother father brother sister son daughter baby friend wife husband boyfriend grandmother grandfather aunt uncle cousin parents".split()
PEOPLE = "teacher student doctor nurse boss boy girl man woman people children".split()
FOOD_MASS = "bread cheese pizza fish meat soup salad cake candy chocolate".split()
FOOD_COUNT = "apple banana egg hamburger sandwich cookie".split()
MEALS = "breakfast lunch dinner".split()
DRINKS = "water coffee tea milk juice beer".split()
ANIMALS = "dog cat bird horse cow pig".split()
THINGS = "car computer phone book pen pencil picture shirt hat dress chair table bed bicycle".split()
THINGS_PL = {"shoes": "shoes", "clothes": "clothes"}
COLORS = "red blue green yellow black white pink orange brown purple".split()
NUMS = {"one": "one", "two": "two", "three": "three", "four": "four", "six": "six", "seven": "seven", "eight": "eight", "nine": "nine"}
PLACE = {"school": "to school", "work": "to work", "home": "home", "hospital": "to the hospital", "doctor": "to the doctor",
         "church": "to church", "library": "to the library", "office": "to the office", "park": "to the park",
         "beach": "to the beach", "restaurant": "to a restaurant", "college": "to college", "bathroom": "to the bathroom",
         "kitchen": "to the kitchen", "city": "to the city", "america": "to America", "party": "to the party",
         "outside": "outside", "bed": "to bed", "store": None}
PLACE = {k: v for k, v in PLACE.items() if v}
PLACE_AT = {"school": "at school", "work": "at work", "home": "at home", "hospital": "in the hospital", "library": "at the library",
            "office": "at the office", "park": "at the park", "beach": "at the beach", "church": "at church",
            "kitchen": "in the kitchen", "bedroom": "in the bedroom", "outside": "outside", "inside": "inside", "college": "in college"}
FEEL = "happy sad angry tired hungry thirsty sick fine good bad ready late busy bored excited scared hurt mad full deaf hearing cold hot".split()
FEEL_EN = {"scared": "scared", "hurt": "hurt", "full": "full", "cold": "cold", "hot": "hot", "mad": "mad"}
THING_ADJ = "big small new old expensive cheap good bad nice fast slow long great hot cold warm cool".split()
TIME_PAST = {"yesterday": "yesterday"}
TIME_FUT = {"tomorrow": "tomorrow", "later": "later", "soon": "soon"}
TIME_DAY = {d: "on " + d.capitalize() for d in "monday tuesday wednesday thursday friday saturday sunday".split()}
TIME_NOW = {"now": "now", "today": "today"}
HABIT = {"always": "always", "sometimes": "sometimes", "often": "often", "never": "never"}
TIME_OF_DAY = {"morning": "in the morning", "afternoon": "in the afternoon", "night": "at night"}

# verb: (3sg, past, ing, stative?)
V = {"eat": ("eats", "ate", "eating", 0), "drink": ("drinks", "drank", "drinking", 0), "cook": ("cooks", "cooked", "cooking", 0),
     "bake": ("bakes", "baked", "baking", 0), "buy": ("buys", "bought", "buying", 0), "sell": ("sells", "sold", "selling", 0),
     "want": ("wants", "wanted", "wanting", 1), "need": ("needs", "needed", "needing", 1), "like": ("likes", "liked", "liking", 1),
     "love": ("loves", "loved", "loving", 1), "hate": ("hates", "hated", "hating", 1), "have": ("has", "had", "having", 1),
     "read": ("reads", "read", "reading", 0), "write": ("writes", "wrote", "writing", 0), "watch": ("watches", "watched", "watching", 0),
     "play": ("plays", "played", "playing", 0), "drive": ("drives", "drove", "driving", 0), "ride": ("rides", "rode", "riding", 0),
     "wash": ("washes", "washed", "washing", 0), "wear": ("wears", "wore", "wearing", 0), "find": ("finds", "found", "finding", 0),
     "lose": ("loses", "lost", "losing", 0), "bring": ("brings", "brought", "bringing", 0), "see": ("sees", "saw", "seeing", 1),
     "meet": ("meets", "met", "meeting", 0), "call": ("calls", "called", "calling", 0), "help": ("helps", "helped", "helping", 0),
     "visit": ("visits", "visited", "visiting", 0), "clean": ("cleans", "cleaned", "cleaning", 0), "open": ("opens", "opened", "opening", 0),
     "close": ("closes", "closed", "closing", 0), "use": ("uses", "used", "using", 0), "teach": ("teaches", "taught", "teaching", 0),
     "learn": ("learns", "learned", "learning", 0), "know": ("knows", "knew", "knowing", 1), "understand": ("understands", "understood", "understanding", 1),
     "go": ("goes", "went", "going", 0), "come": ("comes", "came", "coming", 0), "walk": ("walks", "walked", "walking", 0),
     "run": ("runs", "ran", "running", 0), "travel": ("travels", "traveled", "traveling", 0), "live": ("lives", "lived", "living", 1),
     "stay": ("stays", "stayed", "staying", 0), "sleep": ("sleeps", "slept", "sleeping", 0), "work": ("works", "worked", "working", 0),
     "study": ("studies", "studied", "studying", 0), "dance": ("dances", "danced", "dancing", 0), "sing": ("sings", "sang", "singing", 0),
     "swim": ("swims", "swam", "swimming", 0), "cry": ("cries", "cried", "crying", 0), "laugh": ("laughs", "laughed", "laughing", 0),
     "sit": ("sits", "sat", "sitting", 0), "stand": ("stands", "stood", "standing", 0), "wait": ("waits", "waited", "waiting", 0),
     "sign": ("signs", "signed", "signing", 0), "leave": ("leaves", "left", "leaving", 0), "arrive": ("arrives", "arrived", "arriving", 0),
     "draw": ("draws", "drew", "drawing", 0), "talk": ("talks", "talked", "talking", 0), "fly": ("flies", "flew", "flying", 0),
     "jump": ("jumps", "jumped", "jumping", 0), "pay": ("pays", "paid", "paying", 0), "fight": ("fights", "fought", "fighting", 0),
     "hug": ("hugs", "hugged", "hugging", 0), "smile": ("smiles", "smiled", "smiling", 0), "worry": ("worries", "worried", "worrying", 0),
     "remember": ("remembers", "remembered", "remembering", 1), "think": ("thinks", "thought", "thinking", 0),
     "listen": ("listens", "listened", "listening", 0), "move": ("moves", "moved", "moving", 0), "text": ("texts", "texted", "texting", 0),
     "email": ("emails", "emailed", "emailing", 0), "start": ("starts", "started", "starting", 0), "stop": ("stops", "stopped", "stopping", 0)}
PP = {"eat": "eaten", "drink": "drunk", "see": "seen", "fly": "flown", "ride": "ridden", "go": "gone", "swim": "swum",
      "write": "written", "meet": "met", "visit": "visited", "drive": "driven", "read": "read", "cook": "cooked", "travel": "traveled",
      "play": "played", "watch": "watched", "bake": "baked", "sing": "sung", "dance": "danced"}
# transitive verb -> object generators
def a(n): return ("an " if n[0] in "aeiou" else "a ") + n
def pl(n):
    irr = {"child": "children", "man": "men", "woman": "women", "person": "people", "fish": "fish", "baby": "babies",
           "dress": "dresses", "bus": "buses", "class": "classes", "sandwich": "sandwiches", "candy": "candies",
           "cookie": "cookies", "party": "parties", "family": "families", "story": "stories", "library": "libraries", "city": "cities"}
    return irr.get(n, n + "s")
POSS = {"me": "my", "you": "your", "he": "his", "we": "our", "they": "their", "np": "the", "np_pl": "their"}

def obj(verb, subj):
    """returns (gloss_tokens, english) for an object suiting `verb`"""
    p = POSS.get(subj, "the")
    c = R.random()
    if verb in ("eat", "cook", "buy", "want", "bring", "bake", "sell"):
        if verb == "bake": n = R.choice(["cake", "bread", "cookie", "pizza"]); return [n], {"cookie": "cookies"}.get(n, n)
        pool = FOOD_MASS + FOOD_COUNT + (MEALS if verb in ("eat", "cook") else []) + (DRINKS if verb in ("want", "buy", "bring") else []) \
            + (THINGS if verb in ("buy", "want", "sell", "bring") else [])
        n = R.choice(pool)
        if n in FOOD_COUNT: return [n], (a(n) if c < .6 else pl(n))
        if n in THINGS: return [n], ((a(n) if c < .6 else "the " + n) if verb != "sell" else p + " " + n)
        return [n], n
    if verb == "drink": n = R.choice(DRINKS); return [n], n
    if verb in ("like", "love", "hate"):
        pool = FOOD_MASS + FOOD_COUNT + ANIMALS + DRINKS + ["music", "school", "movie", "game", "snow", "rain", "winter", "summer"]
        pool = [x for x in pool if x not in ("winter", "summer")]
        if verb == "love" and c < .4: n = R.choice(FAMILY + ["family"]); return [p if subj != "me" else "my", n], ("my" if subj == "me" else p) + " " + n
        n = R.choice(pool)
        return [n], (pl(n) if n in FOOD_COUNT + ANIMALS + ["movie", "game"] else n)
    if verb == "have":
        if c < .4:
            k, num = R.choice(list(NUMS.items())); n = R.choice(ANIMALS + ["brother", "sister", "son", "daughter", "cousin", "car", "book", "child"])
            return [k, n], (a(n) if k == "one" else num + " " + pl(n))
        n = R.choice(ANIMALS + THINGS + ["problem", "idea", "money", "time", "class"])
        return [n], (n if n in ("money", "time") else a(n))
    if verb in ("read",): n = R.choice(["book", "email", "paper", "story", "text"]); return [n], a(n) if n != "paper" else "the paper"
    if verb == "write": n = R.choice(["email", "story", "paper", "book", "name"]); return [n], (a(n) if n != "name" else p + " name")
    if verb == "watch": n = R.choice(["movie", "tv", "game", "dog", "children"]); return [n], {"movie": "a movie", "tv": "TV", "game": "the game", "dog": "the dog", "children": "the children"}[n]
    if verb == "play": n = R.choice(["game", "music", "computer"]); return [n], {"game": "a game", "music": "music", "computer": "on the computer"}[n]
    if verb == "drive": n = R.choice(["car", "bus"]); return [n], ("a " if c < .5 else p + " ") + n
    if verb == "ride": n = R.choice(["bicycle", "horse", "bus", "train", "airplane"]); return [n], ("the " if n in ("bus", "train") else "a ") + n
    if verb in ("wash", "wear", "find", "lose", "use", "clean", "pay"):
        pool = {"wash": ["car", "clothes", "shirt", "dog"], "wear": ["shirt", "hat", "dress", "shoes", "clothes"],
                "find": ["phone", "book", "pen", "dog", "cat", "car", "money"], "lose": ["phone", "book", "pen", "money", "game"],
                "use": ["computer", "phone", "pen", "pencil", "internet"], "clean": ["house", "room", "kitchen", "car", "bedroom", "table"],
                "pay": ["money", "doctor", "teacher"]}[verb]
        n = R.choice(pool)
        if verb == "pay": return [n], ("the " + n) if n != "money" else "the money"
        if verb == "lose" and n == "game": return [n], "the game"
        if verb == "use" and n == "internet": return [n], "the internet"
        if verb == "clean" or (verb == "wash" and n == "car" and c < .5): return [n], "the " + n
        return [n], p + " " + n
    if verb in ("see", "meet", "call", "help", "visit", "teach", "hug"):
        if verb == "teach" and c < .5: n = R.choice(["student", "children", "class", "english", "sign_language"]); return [n], {"student": "the students", "children": "the children", "class": "the class", "english": "English", "sign_language": "sign language"}[n]
        if c < .45 or verb == "hug":
            n = R.choice(FAMILY); return [n] if subj == "me" and c < .25 else [p if subj != "me" else "my", n], ("my" if subj == "me" else p) + " " + n
        if c < .7 and subj != "you": return ["you"], "you"
        n = R.choice(["friend", "teacher", "doctor", "boss", "student"]); return [n], ("a " if n == "friend" else "the ") + n
    if verb in ("learn", "study"): n = R.choice(["sign_language", "english", "math"] if verb == "study" else ["sign_language", "english", "sign"]); n = "english" if n == "math" else n; return [n], {"sign_language": "sign language", "english": "English", "sign": "to sign"}[n]
    if verb in ("open", "close"): n = R.choice(["door", "window", "book", "computer"]); return [n], "the " + n
    if verb in ("know", "understand", "remember"):
        if c < .5: return [], ""
        n = R.choice(["you", "teacher", "name", "story", "answer"]); return [n], {"you": "you", "teacher": "the teacher", "name": p + " name", "story": "the story", "answer": "the answer"}[n]
    if verb in ("text", "email"): n = R.choice(FAMILY[:4] + ["friend", "teacher", "boss"]); return [n], ("my" if subj == "me" else p) + " " + n
    return [], ""

TRANS = ["eat", "drink", "cook", "bake", "buy", "want", "need", "like", "love", "hate", "have", "read", "write", "watch", "play",
         "drive", "ride", "wash", "wear", "find", "lose", "bring", "see", "meet", "call", "help", "visit", "clean", "open", "close",
         "use", "teach", "learn", "study", "know", "understand", "remember", "text", "email", "pay", "hug", "sell"]
INTRANS = ["sleep", "work", "study", "dance", "sing", "swim", "cry", "laugh", "sit", "stand", "wait", "sign", "cook", "read",
           "walk", "run", "draw", "talk", "jump", "smile", "worry", "listen", "play", "travel", "leave", "arrive", "think", "move"]

SUBJ = {"me": ("I", "1s"), "you": ("you", "2"), "he": ("he", "3s"), "we": ("we", "pl"), "they": ("they", "pl")}
def np_subj():
    c = R.random()
    if c < .35: return ["me"], "I", "1s", "me"
    if c < .5: return ["you"], "you", "2", "you"
    if c < .6: return ["he"], "he", "3s", "he"
    if c < .7: return ["we"], "we", "pl", "we"
    if c < .75: return ["they"], "they", "pl", "they"
    n = R.choice(FAMILY + PEOPLE)
    per = "pl" if n in ("people", "children", "parents") else "3s"
    pro = "np_pl" if per == "pl" else "np"           # noun subject: no gendered pronoun guesses downstream
    if n in FAMILY or n in ("teacher", "boss", "doctor"):
        poss = R.choice(["my", "my", "your", ""])
        if poss: return [poss, n], f"{poss} {n}", per, pro
        if n in FAMILY: return [n], f"my {n}", per, pro     # bare family sign = "my X" in conversation
    det = "" if n in ("people", "children") else "the "
    return [n], det + n, per, pro

BE = {"1s": ("am", "was"), "2": ("are", "were"), "3s": ("is", "was"), "pl": ("are", "were")}
DO = {"1s": "do", "2": "do", "3s": "does", "pl": "do"}

def conj(v, per, tense, neg=False):
    s3, past, ing, stative = V[v]
    if tense == "past":  return ("did not " + v) if neg else past
    if tense == "fut":   return ("will not " if neg else "will ") + v
    if tense == "prog":  return f"{BE[per][0]}{' not' if neg else ''} {ing}"
    if tense == "perf_never": return ("have never " if per != "3s" else "has never ") + PP.get(v, past)
    if neg: return f"{'does' if per == '3s' else 'do'} not {v}"
    return s3 if per == "3s" else v

def time_frame():
    """returns (gloss_tokens_front, english_suffix_or_prefix, tense_kind, position)"""
    c = R.random()
    if c < .30: return [], "", None
    if c < .42: k = "yesterday"; return [k], "yesterday", "past"
    if c < .56: k, e = R.choice(list(TIME_FUT.items())); return [k], e, "fut"
    if c < .64: k, e = R.choice(list(TIME_DAY.items())); return [k], e, "fut" if R.random() < .5 else "past"
    if c < .76: k, e = R.choice(list(TIME_NOW.items())); return [k], e, "now"
    if c < .88: k, e = R.choice(list(TIME_OF_DAY.items())); return [k], e, "habit"
    k, e = R.choice(list(HABIT.items())); return [k], e, "adv"

def cap(s): s = s.strip(); return s[0].upper() + s[1:]
def fin(s, q=False): s = " ".join(s.split()); return cap(s).rstrip(".?") + ("?" if q else ".")

def sent_action():
    sg, se, per, pro = np_subj()
    trans = R.random() < .7
    v = R.choice(TRANS if trans else INTRANS)
    og, oe = obj(v, pro) if trans else ([], "")
    if v in ("know", "understand", "remember") and not og: pass
    tg, te, tk = time_frame()
    neg = R.random() < .15 and tk not in ("adv",)
    stative = V[v][3]
    tense = {"past": "past", "fut": "fut", "now": "prog" if not stative else "pres", "habit": "pres", None: "pres" if stative else "prog", "adv": "pres"}[tk]
    if tk == "adv" and te == "never" and R.random() < .4 and v in PP and per != "3s": tense = "perf_never"
    verb = conj(v, per, tense, neg)
    if tk == "adv":
        if tense == "perf_never": vp = verb
        else: vp = f"{te} {verb}"
        gloss = sg + [tg[0]] + [v] + og
    else:
        vp = verb
        gloss = tg + sg + (["not"] if neg else []) + [v] + og
    eng = f"{se} {vp} {oe}" + (f" {te}" if tk not in ("adv", None) else "")
    is_q = sg == ["you"] and not neg and R.random() < .85 and tk != "adv"
    if is_q:  # yes/no question
        if tense == "prog": eng = f"are you {V[v][2]} {oe}" + (f" {te}" if te and tk != "adv" else "")
        elif tense == "past": eng = f"did you {v} {oe} {te}"
        elif tense == "fut": eng = f"will you {v} {oe} {te}"
        else: eng = f"do you {v} {oe}" + (f" {te}" if tk == "habit" else "")
        return gloss, fin(eng, True)
    return gloss, fin(eng)

def sent_state():
    sg, se, per, pro = np_subj()
    adj = R.choice(FEEL)
    tg, te, tk = time_frame()
    if tk == "adv": tg, te, tk = [], "", None
    neg = R.random() < .15
    tense = "past" if tk == "past" else ("fut" if tk == "fut" else "pres")
    be = {"past": BE[per][1], "fut": "will be", "pres": BE[per][0]}[tense]
    if neg: be = be.replace("will be", "will not be") if tense == "fut" else be + " not"
    gloss = tg + sg + (["not"] if neg else []) + [adj]
    if sg == ["you"] and not neg and R.random() < .8:
        q = {"past": "were you", "fut": "will you be", "pres": "are you"}[tense]
        return gloss, fin(f"{q} {adj} {te}", True)
    return gloss, fin(f"{se} {be} {adj} {te}")

def sent_thing():
    c = R.random()
    if c < .5:
        n = R.choice(THINGS + ANIMALS + ["house", "room", "school", "food", "weather", "class", "movie", "game", "party", "book"])
        adj = R.choice(THING_ADJ + COLORS)
        poss = R.choice(["my", "your", "", ""])
        g = ([poss] if poss else []) + [n, adj]
        e = f"{poss or 'the'} {n} is {adj}"
        if poss == "your" and R.random() < .5: return g, fin(f"is your {n} {adj}", True)
        return g, fin(e)
    if c < .7:
        k = R.choice(["weather", "today", "tomorrow", "yesterday"]); adj = R.choice(["cold", "hot", "warm", "cool", "nice", "bad", "good"])
        e = {"weather": f"the weather is {adj}", "today": f"it is {adj} today", "tomorrow": f"it will be {adj} tomorrow", "yesterday": f"it was {adj} yesterday"}[k]
        return [k, adj], fin(e)
    n = R.choice(["rain", "snow"]); k = R.choice(["today", "tomorrow", "yesterday", "now"])
    e = {"today": f"it is {n}ing today", "tomorrow": f"it will {n} tomorrow", "yesterday": f"it {n}ed yesterday", "now": f"it is {n}ing now"}[k]
    return [k, n], fin(e)

def sent_wh():
    c = R.random(); wh_first = R.random() < .25
    if c < .25:
        v = R.choice(["want", "eat", "drink", "like", "need", "buy", "cook", "read", "watch", "study", "learn"])
        sg, se, per, pro = R.choice([(["you"], "you", "2", "you"), (["he"], "he", "3s", "he"), (["your", "mother"], "your mother", "3s", "he"), (["they"], "they", "pl", "they")])
        tg, te, tk = R.choice([([], "", None), (["yesterday"], "yesterday", "past"), (["tomorrow"], "tomorrow", "fut"), (["now"], "now", "now")])
        aux = {"past": "did", "fut": "will", "now": BE[per][0], None: DO[per]}[tk]
        verb = V[v][2] if tk == "now" else v
        g = tg + (["what"] + sg + [v] if wh_first else sg + [v, "what"])
        return g, fin(f"what {aux} {se} {verb} {te}", True)
    if c < .5:
        v, place = R.choice([("live", None), ("work", None), ("go", None), ("study", None), ("sleep", None), ("eat", None)])
        sg, se, per, pro = R.choice([(["you"], "you", "2", "you"), (["he"], "he", "3s", "he"), (["your", "friend"], "your friend", "3s", "he"), (["your", "family"], "your family", "3s", "he")])
        tg, te, tk = R.choice([([], "", None), (["yesterday"], "yesterday", "past"), (["tomorrow"], "tomorrow", "fut")])
        if v == "go" and tk is None: tk = "now"
        aux = {"past": "did", "fut": "will", "now": BE[per][0], None: DO[per]}[tk]
        verb = V[v][2] if tk == "now" else v
        g = tg + (["where"] + sg + [v] if wh_first else sg + [v, "where"])
        return g, fin(f"where {aux} {se} {verb} {te}", True)
    if c < .62:
        n = R.choice(["bathroom", "phone", "car", "school", "doctor", "teacher", "book", "dog", "cat", "kitchen", "party", "restaurant", "library", "office", "hospital"])
        poss = R.choice(["", "my", "your"])
        g = ([poss] if poss else []) + [n]
        return (["where"] + g if wh_first else g + ["where"]), fin(f"where is {poss or 'the'} {n}", True)
    if c < .72:
        n = R.choice(["party", "class", "dinner", "lunch", "movie", "game", "birthday", "meeting"])
        n = "party" if n == "meeting" else n
        poss = "your " if n == "birthday" else "the "
        return (["when", n] if wh_first else [n, "when"]) if n != "birthday" else (["your", n, "when"]), fin(f"when is {poss}{n}", True)
    if c < .8:
        adj = R.choice(["sad", "angry", "happy", "tired", "late", "mad", "scared", "excited"])
        sg, se, per = R.choice([(["you"], "you", "2"), (["he"], "he", "3s"), (["they"], "they", "pl")])
        return (["why"] + sg + [adj] if wh_first else sg + [adj, "why"]), fin(f"why {BE[per][0]} {se} {adj}", True)
    if c < .88:
        n = R.choice(["teacher", "boss", "doctor", "friend", "mother", "father"])
        return (["who", "your", n] if wh_first else ["your", n, "who"]), fin(f"who is your {n}", True)
    if c < .94:
        k = R.choice(["feel", "learn_sign"])
        if k == "feel": return (["how", "you", "feel"] if wh_first else ["you", "feel", "how"]), "How do you feel?"
        return (["you", "learn", "sign", "how"]), "How did you learn to sign?"
    n = R.choice(["color", "car", "food", "movie", "class", "game", "book"])
    return (["which", n, "you", "like"] if wh_first else ["you", "like", "which", n]), fin(f"which {n} do you like", True)

def sent_modal():
    sg, se, per, pro = np_subj()
    c = R.random()
    if c < .5:  # can
        v = R.choice(["sign", "help", "drive", "swim", "cook", "dance", "sing", "come", "read", "write", "understand", "go", "play", "ride", "fly"])
        og, oe = obj(v, pro) if v in TRANS and R.random() < .5 else ([], "")
        if v == "help" and not og: og, oe = (["you"], "you") if pro != "you" else (["me"], "me")
        neg = R.random() < .3
        g = sg + ["can"] + (["not"] if neg else []) + [v] + og
        if sg == ["you"] and not neg: return g, fin(f"can you {v} {oe}", True)
        return g, fin(f"{se} {'cannot' if neg else 'can'} {v} {oe}")
    if c < .8:  # want to V
        v = R.choice(["eat", "drink", "go", "learn", "play", "sleep", "watch", "buy", "visit", "travel", "dance", "meet", "read", "cook"])
        if v in ("go", "travel"):
            p = R.choice(list(PLACE.items()) + [("america", "to America")]); og, oe = [p[0]], p[1]
        else: og, oe = obj(v, pro) if v in TRANS else ([], "")
        g = sg + ["want", v] + og
        if sg == ["you"]: return g, fin(f"do you want to {v} {oe}", True)
        return g, fin(f"{se} {V['want'][0] if per == '3s' else 'want'} to {v} {oe}")
    # need
    n = R.choice(["help", "doctor", "water", "money", "time", "car", "phone", "computer", "pen", "rest"])
    n = "help" if n == "rest" else n
    g = sg + ["need", n]
    oe = {"help": "help", "water": "water", "money": "money", "time": "time"}.get(n, a(n))
    if sg == ["you"]: return g, fin(f"do you need {oe}", True)
    return g, fin(f"{se} {'needs' if per == '3s' else 'need'} {oe}")

def sent_motion():
    sg, se, per, pro = np_subj()
    v = R.choice(["go", "go", "come", "walk", "drive", "travel", "stay", "live", "arrive", "run"])
    if v in ("stay", "live"):
        k = R.choice(list(PLACE_AT.items()) + ([("america", "in America"), ("city", "in the city")] if v == "live" else []))
    else:
        k = R.choice(list(PLACE.items()))
        if v == "arrive": k = (k[0], k[1].replace("to the", "at the").replace("to ", "at ").replace("home", "home"))
        if v == "come" and k[0] == "home": pass
    tg, te, tk = time_frame()
    if tk in ("adv",): tg, te, tk = [], "", None
    neg = R.random() < .1
    stative = v == "live"
    tense = {"past": "past", "fut": "fut", "now": "prog", "habit": "pres", None: "pres" if stative else "prog"}[tk]
    gloss = tg + sg + (["not"] if neg else []) + [v, k[0]]
    if sg == ["you"] and not neg and R.random() < .8:
        if tense == "prog": e = f"are you {V[v][2]} {k[1]} {te}"
        elif tense == "past": e = f"did you {v} {k[1]} {te}"
        elif tense == "fut": e = f"will you {v} {k[1]} {te}"
        else: e = f"do you {v} {k[1]} {te}"
        return gloss, fin(e, True)
    return gloss, fin(f"{se} {conj(v, per, tense, neg)} {k[1]} {te}")

FIXED = [("hello", "Hello."), ("hi", None), ("thank_you", "Thank you."), ("bye", "Goodbye."), ("sorry", "I'm sorry."),
         ("welcome", "You're welcome."), ("please", "Please."), ("yes", "Yes."), ("no", "No."), ("maybe", "Maybe."),
         ("hello how you", "Hello, how are you?"), ("nice meet you", "Nice to meet you."), ("thank_you you", "Thank you."),
         ("me fine thank_you", "I'm fine, thank you."), ("good morning", "Good morning."), ("good night", "Good night."),
         ("see you later", "See you later."), ("see you tomorrow", "See you tomorrow."), ("me sorry", "I'm sorry."),
         ("please help", "Please help."), ("help", "Help!"), ("wait", "Wait."), ("stop", "Stop."), ("again please", "Again, please."),
         ("sign again please", "Please sign that again."), ("me deaf you", "I'm deaf, and you?"), ("happy birthday", "Happy birthday!"),
         ("thank_you very much", None), ("me love you", "I love you."), ("bye see you", "Bye, see you."),
         ("alright", "Alright."), ("great", "Great!"), ("me understand", "I understand."), ("me not know", "I don't know."),
         ("me know", "I know."), ("me think", "I think so."), ("finish", "Finished."), ("me finish", "I'm finished."),
         ("you finish", "Are you finished?"), ("me ready", "I'm ready."), ("you ready", "Are you ready?"),
         ("welcome home", "Welcome home."), ("happy holiday", "Happy holidays!"), ("merry christmas", None),
         ("good idea", "Good idea."), ("no problem", "No problem."), ("same", "Same."), ("me same", "Me too.")]

def sent_please():
    v = R.choice(["help", "wait", "come", "sit", "stand", "sign", "open", "close", "call", "bring", "stop", "listen", "eat", "drink", "read", "write"])
    og, oe = obj(v, "you") if v in TRANS and R.random() < .7 else ([], "")
    if v == "help": og, oe = (["me"], "me") if R.random() < .7 else (og, oe)
    if v == "call": og, oe = (["me"], "me") if R.random() < .5 else (og, oe)
    tail = R.choice([[], ["now"], ["later"]]) if v not in ("help",) else []
    g = ["please", v] + og + tail if R.random() < .7 else [v] + og + tail + ["please"]
    return g, fin(f"please {v} {oe} {' '.join(tail)}")

def sent_have_family():
    sg, se, per, pro = R.choice([(["me"], "I", "1s", "me"), (["you"], "you", "2", "you"), (["he"], "he", "3s", "he"), (["we"], "we", "pl", "we")])
    k, num = R.choice(list(NUMS.items())); n = R.choice(["brother", "sister", "son", "daughter", "child", "cousin", "dog", "cat", "car", "friend"])
    noun = a(n) if k == "one" else f"{num} {pl(n)}"
    g = sg + ["have", k, n]
    if sg == ["you"]: return g, fin(f"do you have {noun}", True)
    return g, fin(f"{se} {'has' if per == '3s' else 'have'} {noun}")

GENS = [(sent_action, 34), (sent_state, 14), (sent_thing, 8), (sent_wh, 16), (sent_modal, 12), (sent_motion, 10), (sent_please, 4), (sent_have_family, 2)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = set(w.strip() for w in open("data/aslcitizen/all_words.txt") if w.strip()) | {"me", "you"}
    test = {r["gloss"].strip() for r in csv.DictReader(open(OUT / "eval_gloss_en.tsv"), delimiter="\t")}
    fns, ws = zip(*GENS); pairs = {}
    bad_vocab = {}
    for _ in range(400000):
        g, e = R.choices(fns, ws)[0]()
        g = [x for x in g if x]
        if not g or not e or len(g) > 8: continue
        miss = [x for x in g if x not in vocab]
        if miss:
            for m in miss: bad_vocab[m] = bad_vocab.get(m, 0) + 1
            continue
        key = " ".join(g)
        if key in test: continue
        pairs.setdefault(key, e)
        if len(pairs) >= 24000: break
    for g, e in FIXED:
        if e and g not in test and all(x in vocab for x in g.split()): pairs[g] = e
    items = sorted(pairs.items()); R.shuffle(items)
    nv = len(items) // 20
    for name, part in (("val", items[:nv]), ("train", items[nv:])):
        with open(OUT / f"{name}.tsv", "w") as f:
            f.write("gloss\tenglish\n"); f.writelines(f"{g}\t{e}\n" for g, e in part)
    print(f"{len(items)} unique pairs -> train {len(items)-nv} / val {nv}; dropped-out-of-vocab words: {sorted(bad_vocab, key=bad_vocab.get, reverse=True)[:15]}")
    for g, e in items[:25]: print(f"  {g.upper():40s} {e}")

if __name__ == "__main__":
    main()
