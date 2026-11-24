"""Faithfulness check for composed sentences: does the English keep every signed word and add no new content words?
Shared by the live composer (safety fallback) and scripts/eval_composer.py."""
import re

class _Lem:
    """WordNet lemmatizer, or a crude suffix stripper if the nltk 'wordnet' data is not installed."""
    def __init__(self):
        try:
            from nltk.stem import WordNetLemmatizer
            self.wn = WordNetLemmatizer(); self.wn.lemmatize("going", "v")
        except Exception:
            self.wn = None
    def lemmatize(self, w, pos):
        if self.wn: return self.wn.lemmatize(w, pos)
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) > len(suf) + 2: return w[: -len(suf)]
        return w

_L = _Lem()
# words the composer may always add (grammar), and gloss -> acceptable surface forms
FUNC = set("""a an the is are am was were be been being do does did done to of and or but so that this these those it its
there their they them then than very just also for in on at by with as from into about up out over have has had will would
shall should may might must not n't 's 're 've 'll 'd m i me my mine you your yours he him his she her we us our it's let let's
what who where why when how which can could some any much many more please today now please yeah okay ok together too
going gone got get am being one""".split())
ALIAS = {"me": {"i", "me", "my", "myself", "mine", "im"}, "you": {"you", "your", "yours"}, "he": {"he", "him", "his"},
         "we": {"we", "us", "our", "let"}, "they": {"they", "them", "their"}, "my": {"my", "i", "me", "mine"},
         "not": {"not", "n't", "dont", "don't", "doesn't", "can't", "cannot", "never", "no"}, "thank_you": {"thank", "thanks"},
         "sign_language": {"sign", "language"}, "hello": {"hello", "hi"}, "bye": {"bye", "goodbye"},
         "finish": {"finish", "finished", "done"}, "have": {"have", "has", "had", "got"}, "time": {"time", "o'clock"},
         "how": {"how"}, "cost": {"cost", "price", "much"}, "can": {"can", "could", "able"}, "two": {"two", "2"}}

_CONTR = [("can't", "can not"), ("won't", "will not"), ("n't", " not"), ("'m", " am"), ("'re", " are"), ("'s", " is"),
          ("'ll", " will"), ("'ve", " have"), ("'d", " would")]

def expand(text):
    text = text.lower().replace("’", "'")
    for a, b in _CONTR: text = text.replace(a, b)
    return text

def lemmas(text):
    out = set()
    for t in re.findall(r"[a-z']+", expand(text)):
        out.add(t)
        for p in "vna": out.add(_L.lemmatize(t.strip("'"), p))
    return out

def score(gloss, out):
    g = gloss.split(); ol = lemmas(out)
    cov = sum(1 for w in g if (ALIAS.get(w, {w}) | {w}) & ol) / len(g)
    allowed = set(FUNC)
    for w in g: allowed |= ALIAS.get(w, {w}) | lemmas(w.replace("_", " "))
    added = [t for t in re.findall(r"[a-z']+", expand(out)) if t not in allowed and not (lemmas(t) & allowed)]
    return cov, added

