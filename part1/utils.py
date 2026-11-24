import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import hashlib
import re
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from functools import lru_cache
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


@lru_cache(maxsize=20000)
def _adjective_synonyms(word):
    """Single-word alternatives from the first WordNet adjective sense."""
    synsets = wordnet.synsets(word, pos=wordnet.ADJ)
    if not synsets:
        return ()
    return tuple(sorted({
        lemma.name().lower()
        for lemma in synsets[0].lemmas()
        if re.fullmatch(r"[A-Za-z]+", lemma.name())
        and lemma.name().lower() != word
    }))


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    text = example["text"]
    seed = hashlib.sha256(("q2-combined-v5:" + text).encode("utf-8")).digest()
    rng = random.Random(seed)

    # ---------- 1. Lexicons ----------
    # Informal slang / abbreviations 
    replacements = {
        "different from": "different than", "better than": "better then",
        "worse than": "worse then", "the effect": "the affect",
        "is not": "ain't", "are not": "ain't",
        "for sure": "fsfs", "for real": "frfr", "oh no": "oh naurrr",
        "very": "hella", "probably": "prolly", "because": "cuz",
        "people": "ppl", "you": "u", "your": "ur", "before": "b4",
        "in my opinion": "imo", "to be honest": "tbh", "by the way": "btw",
        "i don't know": "idk", "i think": "i reckon",
        "kind of": "kinda", "sort of": "sorta", "a lot of": "lots of",
        "going to": "gonna", "want to": "wanna", "have to": "gotta",
        "out of": "outta", "all right": "alright", "tonight": "2nite",
        "though": "tho", "through": "thru", "please": "pls",
        "something": "smth", "someone": "some1", "anyone": "any1",
        "everyone": "every1", "about": "abt", "really": "rly",
        "especially": "esp", "definitely": "def", "seriously": "srsly",
        "totally": "totes",
    }

    # Real-word typos: the misspelling is a different valid word 
    confusables = {
        "definitely": "defiantly", "casual": "causal", "effect": "affect",
        "affect": "effect", "than": "then", "then": "than",
        "their": "there", "there": "their", "great": "grate",
        "poor": "pore", "quite": "quiet", "lose": "loose", "loose": "lose",
        "plot": "plod", "scene": "seen", "scenes": "seens",
        "acting": "acing", "actor": "actir", "whole": "hole",
        "weak": "week", "waste": "waist", "mind": "mined",
        "bored": "board", "cast": "cat", "fair": "fare",
    }

    # Readable phonetic misspellings of sentiment words
    phonetic = {
        "wonderful": "wunderful", "wonderfully": "wunderfully",
        "excellent": "excelent", "amazing": "amazin", "awesome": "awsome",
        "brilliant": "briliant", "beautiful": "beutiful", "perfect": "perfct",
        "magnificent": "magnificant", "fantastic": "fantstic",
        "enjoyable": "enjoyble", "enjoyed": "enjoyd", "loved": "luved",
        "love": "luv", "best": "bst", "good": "gud", "funny": "funy",
        "favorite": "favorit", "recommend": "recomend", "worst": "wrst",
        "terrible": "terible", "awful": "awfull", "horrible": "horible",
        "boring": "borring", "disappointing": "dissapointing",
        "disappointed": "dissapointed", "disappointment": "dissapointment",
        "stupid": "stupd", "annoying": "anoying", "pathetic": "pathtic",
        "mediocre": "mediocer", "ridiculous": "rediculous", "dull": "dul",
        "hate": "hait", "hated": "haited", "worse": "wors",
        "worthless": "worthles", "predictable": "predictble",
    }

    sentiment = set(phonetic) | set(confusables) | {
        "fun", "superb", "touching", "moving", "powerful", "hilarious",
        "masterpiece", "gem", "engrossing", "engrossed", "surprised",
        "charming", "delightful", "stunning", "outstanding", "solid",
        "gripping", "entertaining", "entertained", "impressive", "classic",
        "gorgeous", "cute", "sweet", "nice", "interesting", "recommended",
        "bland", "lame", "mess", "tedious", "forgettable", "unwatchable",
        "dreadful", "painful", "pointless", "cheesy", "cliched", "silly",
        "lousy", "garbage", "trash", "crap", "rubbish", "avoid", "skip",
        "instead", "failed", "fails", "mindless", "confusing", "unfunny",
        "uninteresting", "overrated", "laughable", "bad",
    }

    protected = {
        "no", "not", "never", "cannot", "neither", "nor", "nobody",
        "nothing", "nowhere", "without", "hardly", "barely", "don", "isn",
        "wasn", "doesn", "didn", "won", "wouldn", "couldn", "shouldn",
    }
    protected.update(
        w.lower() for r in replacements.values() for w in re.findall(r"[A-Za-z]+", r)
    )

    def match_case(original, new):
        if original.isupper() and len(original) > 1:
            return new.upper()
        if original[0].isupper():
            return new[0].upper() + new[1:]
        return new

    # ---------- 2. WordNet adjective synonyms 
    words = [
        m for m in re.finditer(r"<[^>]*>|(?<![\w'’])[A-Za-z]+(?![\w'’])", text)
        if not m.group(0).startswith("<")
    ]
    candidates = []
    for m, (_, tag) in zip(words, pos_tag([w.group(0) for w in words])):
        low = m.group(0).lower()
        if tag == "JJ" and low not in sentiment and low not in protected:
            syns = [s for s in _adjective_synonyms(low) if len(s) <= len(low) + 4]
            if syns:
                candidates.append((m, syns))
    rng.shuffle(candidates)
    chosen = sorted(candidates[:2], key=lambda c: c[0].start(), reverse=True)
    for m, syns in chosen:  # right-to-left keeps offsets valid
        text = text[:m.start()] + match_case(m.group(0), rng.choice(syns)) + text[m.end():]

    # ---------- 3. Slang / phrase replacement ----------
    phrases = sorted(replacements, key=lambda p: (-len(p), p))
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(p) for p in phrases) + r")(?!\w)",
        flags=re.IGNORECASE,
    )

    def replace_match(m):
        original = m.group(0)
        if rng.random() >= 0.5:
            return original
        return match_case(original, replacements[original.lower()])

    transformed = pattern.sub(replace_match, text)

    # ---------- 4. Typos ----------
    near = {"a": "qwsz", "e": "wsdr", "i": "jkl", "o": "klp", "u": "yhj"}
    leet = {"o": "0", "i": "1", "e": "3", "a": "4"}

    def perturb(word, strong):
        n = len(word)
        modes = ["vowel", "leet"] if n < 4 else ["swap", "drop", "double", "vowel", "leet"]
        mode = rng.choice(modes)
        if mode == "swap":
            i = rng.randrange(1, n - 2)  # inner letters only
            if word[i].lower() != word[i + 1].lower():
                return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        elif mode == "drop":
            i = rng.randrange(1, n - 1)
            return word[:i] + word[i + 1:]
        elif mode == "double":
            i = rng.randrange(1, n - 1)
            return word[:i] + word[i] + word[i:]
        elif mode == "leet" and strong:
            idx = [i for i, c in enumerate(word) if c.lower() in leet]
            if idx:
                i = rng.choice(idx)
                return word[:i] + leet[word[i].lower()] + word[i + 1:]
        vowels = [i for i, c in enumerate(word) if c.lower() in near]
        if vowels:
            i = rng.choice(vowels)
            rep = rng.choice(near[word[i].lower()])
            if word[i].isupper():
                rep = rep.upper()
            return word[:i] + rep + word[i + 1:]
        return word

    def edit(m):
        word = m.group(0)
        low = word.lower()
        if word.startswith("<") or low in protected:
            return word
        if low in confusables and rng.random() < 0.9:      # real-word typo
            return match_case(word, confusables[low])
        if low in phonetic and rng.random() < 0.8:         # phonetic sentiment typo
            return match_case(word, phonetic[low])
        if low in sentiment:                               # other sentiment words
            return perturb(word, True) if rng.random() < 0.8 else word
        if len(word) >= 4 and rng.random() < 0.4:          # generic typo
            return perturb(word, False)
        return word

    example["text"] = re.sub(
        r"<[^>]*>|(?<![\w'’])[A-Za-z]{3,}(?![\w'’])", edit, transformed
    )

    ##### YOUR CODE ENDS HERE ######

    return example
