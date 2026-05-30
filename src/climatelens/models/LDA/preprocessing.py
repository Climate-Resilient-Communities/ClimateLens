import gensim
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("stopwords")


def isEnglish(s: str) -> bool:
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def fix_text(txt: str) -> str:
    if not isEnglish(txt):
        for i, s in enumerate(txt):
            if not isEnglish(s):
                if len(txt) >= i + 2:
                    if txt[i + 1] == "s":
                        txt = txt.replace(s, "'")
                    elif txt[i + 1] == " " and "'" not in txt:
                        txt = txt.replace(s, "-")
                    else:
                        txt = txt.replace(s, "")
                else:
                    txt = txt.replace(s, "")
    return txt


additional_stopwords: list[str] = [
    # platform-specific
    "rt", "tweet", "repost", "replied", "comments", "comment",
    "upvote", "downvote", "subreddit", "thread", "user", "followers",
    "post", "share", "like", "reply", "hashtag", "hashtags", "link",
    "bio", "mention", "tagged", "followed", "following", "message", "profile",
    # filler and vague
    "literally", "actually", "kinda", "totally", "idk", "btw", "omg",
    "lol", "lmao", "you know", "basically", "seriously", "honestly",
    "obviously", "probably", "idc", "maybe", "exactly", "apparently",
    "definitely", "pretty", "thing", "stuff", "I mean", "uh", "um",
    "err", "ah", "yeah", "okay", "gotcha", "bruh", "bro", "yup",
    "nope", "nah", "wait", "pretty much", "as if", "thank you",
    "thanks", "for sure", "in fact", "kind of", "sort of", "at least",
    "like I said", "at the end of the day",
    # common, redundant
    "it is", "there is", "there are", "that's", "this is", "the thing",
    "of course", "at the moment", "you know what I mean", "don't know",
    "do you know", "all the time", "just saying", "honestly speaking",
    "the truth is", "somebody", "situation", "matter", "fact", "time",
    "point", "problem", "issue", "question", "extremely", "super",
    "highly", "completely", "entirely",
    # found after initial topic modeling
    "think", "talk", "look", "tell", "happen", "increase", "years",
    "people", "sure", "know", "come", "today", "year", "earth", "world",
]

climate_terms: list[str] = ["climate", "change", "climate change", "global warming"]
profanities: list[str] = ["fuck", "shit", "ass"]

stopwords: list[str] = list(gensim.parsing.preprocessing.STOPWORDS)
for word in additional_stopwords + climate_terms + profanities:
    if word not in stopwords:
        stopwords.append(word)


def lemmatize(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text, pos="v")


def preprocess(text: str, stopwords: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    result: list[str] = []
    stem_dict: list[tuple[str, str]] = []

    text = fix_text(text)
    tokens = gensim.utils.simple_preprocess(text)

    for token in tokens:
        if token not in stopwords:
            lemmatized_token = lemmatize(token)
            if lemmatized_token not in stopwords and len(lemmatized_token) > 3:
                result.append(lemmatized_token)
                stem_dict.append((lemmatized_token, token))

    return result, stem_dict