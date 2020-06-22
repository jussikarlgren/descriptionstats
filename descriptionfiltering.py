import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from random import sample

datadirectory = "/Users/jik/Box/Spotify-Podcasts-2020/podcasts-no-audio-13GB/"
metadatafile = "metadata.tsv"
outputdirectory = "/Users/jik/data/trec-podcasts/"
outfile = "sampledescriptions-many-of-them.tsv"

selfsimilaritythreshold = 0.4
acrossepisodesthreshold = 0.5
samplesize = 1000000
numberofoutputitems = 100000
# length given in characters
minlength = 20
maxlength = 750


def takeoutboilerplate(description: str):
    try:
        r = description\
            .replace("This episode is sponsored by", "")\
            .replace("Support this podcast:", "")\
            .replace("Anchor: The easiest way to make a podcast", "")\
            .replace("https://anchor.fm/app", "")\
            .replace("https://anchor.fm/reseller/support", "")\
            .replace("Send in a voice message:", "")\
            .replace(" https://anchor.fm/theleafsconvo/message", "")\
            .replace("https://anchor.fm/theleafsconvo/suppor", "")
    except:
        # if snafu just return what we came in with ok
        r = description
    return r


def readandfiltermetadatafile(datafilename: str):
    """
    reads tsv file with lines that are metadata records as per below
    """
    # 0-1 show_uri, show_name,
    # 2   show_description,
    # 3-4 publisher,language,
    # 5-7 rss_link,episode_uri,episode_name,
    # 8   episode_description,
    # 9-11 duration,show_filename_prefix,episode_filename_prefix

    showposition = 2
    episodeposition = 8
    episodenameposition = 7
    items = []
    allofthem = []
    describe = True
    bign = 0
    wronglength = 0
    with open(datafilename, 'r') as infile:
        reader = csv.reader(infile, delimiter="\t")
        for row in reader:
            bign += 1
            record = list(row)
            description = takeoutboilerplate(record[episodeposition])
            cl = len(description)
            # remove the long ones, allow short ones
            # removes just under a fifth of the 100k sample
            if describe:
                allofthem.append(record)
            if minlength < cl < maxlength:
                items.append(record)
                wronglength += 1
    # pick out a subset to play with
    if describe:
        print(f"""Length filter: {wronglength} / {bign}  {wronglength / bign} ({minlength} < length < {maxlength})""")
    if samplesize < len(items):
        candidates = sample(items, samplesize)
    elif describe:
        candidates = allofthem
    else:
        candidates = items
    # remove episodes where the description is too similar to some other episode description
    # removes about 10-12% of the remaining episodes
    retained = []
    alldescriptions = []
    c = 0
    ii = 0
    for oneitem in candidates:
        c += 1
        alldescriptions.append(takeoutboilerplate(oneitem[episodeposition]))
        if c % 2000 == 0:
            wordsimilaritycalculator = TfidfVectorizer(min_df=1, stop_words="english")
            tfidfmatrix = wordsimilaritycalculator.fit_transform(alldescriptions)
            similarity = (tfidfmatrix * tfidfmatrix.T).toarray()
            for corrs in similarity:
                # if the second highest item in the correlation matrix is under the threshold, it's a keeper
                if sorted(list(corrs), reverse=True)[1] < acrossepisodesthreshold:
                    retained.append(candidates[ii])
                    ii += 1
            alldescriptions = []
    # remove such shows where the episode description is too similar to the show description
    # removes another 7% of the items, two thirds of original remaining after this
    if describe:
        print(f"""Cross-similarity filter: {ii} / {len(candidates)} {ii / len(candidates)} ({acrossepisodesthreshold})""")
        retained = allofthem
    reretained = []
    wordsimilaritycalculator = TfidfVectorizer(min_df=1, stop_words="english")
    iii = 0
    for oneitem in retained:
        textstocheck = [oneitem[episodeposition], oneitem[episodenameposition], oneitem[showposition]]
        try:
            tfidfmatrix = wordsimilaritycalculator.fit_transform(textstocheck)
            similarity = (tfidfmatrix * tfidfmatrix.T).toarray()
            alldescriptions.append(oneitem[episodeposition])
            if similarity[0][2] < selfsimilaritythreshold:
                reretained.append(oneitem)
                iii += 1
        except ValueError:
            pass
    if describe:
        print(f"""Self-similarity filter: {iii} / {len(retained)} {iii / len(retained)} ({selfsimilaritythreshold})""")
    opposite = False
    iii = 0
    if opposite:
        reretainedopposite = []
        for i in candidates:
            if i not in reretained:
                reretainedopposite.append(i)
        reretained = reretainedopposite
        print(len(reretained))
    # keep some nice number of items from the set
    if numberofoutputitems < len(reretained):
        delivery = sample(reretained, numberofoutputitems)
    elif describe:
        delivery = []
    else:
        delivery = reretained
    print(len(delivery))
    return delivery


def writedatatotsv(delivery: list, outputdatafilename: str):
    with open(outputdatafilename, "w") as outfile:
        outputter = csv.writer(outfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for oneitem in delivery:
            outputter.writerow(oneitem)


if __name__ == '__main__':
    subsetitems = readandfiltermetadatafile(datadirectory+metadatafile)
    if len(subsetitems) > 0:
        writedatatotsv(subsetitems, outputdirectory+"opposite."+outfile)
