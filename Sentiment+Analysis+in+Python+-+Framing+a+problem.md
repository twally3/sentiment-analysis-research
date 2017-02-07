
# Sentiment Analysis in Python - Framing a problem
By Max Taylor

This is an overview of the full process of developing a Neural Network for sentiment analysis. We will address: How to develop and impement a predictive theory, prepare data, build the network, reduce noise and optimise. This will involve how to attack and solve the problem; Which can be applied throughout future networks.

### Contents
- Loading the dataset
- predictive theory
- Theory Validation
- Preparing input and output data
- Building the network
- Identifying Neural Noise
- Reducing Neural Noise
- Analysing inefficiencies in the network
- Optimising inefficiencies in the network
- Further noise reduction

## Loading the dataset


```python
g = open('reviews.txt','r')
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r')
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()
```


```python
len(reviews)
```




    25000




```python
reviews[0]
```




    'bromwell high is a cartoon comedy . it ran at the same time as some other programs about school life  such as  teachers  . my   years in the teaching profession lead me to believe that bromwell high  s satire is much closer to reality than is  teachers  . the scramble to survive financially  the insightful students who can see right through their pathetic teachers  pomp  the pettiness of the whole situation  all remind me of the schools i knew and their students . when i saw the episode in which a student repeatedly tried to burn down the school  i immediately recalled . . . . . . . . . at . . . . . . . . . . high . a classic line inspector i  m here to sack one of your teachers . student welcome to bromwell high . i expect that many adults of my age think that bromwell high is far fetched . what a pity that it isn  t   '




```python
labels[0]
```




    'POSITIVE'



## Develop a predictive theory

Look over the data and consider the best way to use the input information to effectively get the output information.


```python
def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")
```


```python
print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)
```

    labels.txt 	 : 	 reviews.txt
    
    NEGATIVE	:	this movie is terrible but it has some good effects .  ...
    POSITIVE	:	adrian pasdar is excellent is this film . he makes a fascinating woman .  ...
    NEGATIVE	:	comment this movie is impossible . is terrible  very improbable  bad interpretat...
    POSITIVE	:	excellent episode movie ala pulp fiction .  days   suicides . it doesnt get more...
    NEGATIVE	:	if you haven  t seen this  it  s terrible . it is pure trash . i saw this about ...
    POSITIVE	:	this schiffer guy is a real genius  the movie is of excellent quality and both e...


In this case, the data should be split into words as they convey the most meaning. For example, single characters do not contain any form of context and whole sentences are too large and general for the network to compute.

## Theory validation

Having produced a predictive theory it can be helpful to validate the theory. In this case it was decided that words were the most helpful to us, so this is what we will test!


```python
# Import dependencies
from collections import Counter
import numpy as np
```


```python
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()
```


```python
for i in range(len(reviews)):
    if labels[i] == 'POSITIVE':
        for word in reviews[i].split(' '):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(' '):
            negative_counts[word] += 1
            total_counts[word] += 1
```


```python
positive_counts.most_common()[0:50]
```




    [('', 550468),
     ('the', 173324),
     ('.', 159654),
     ('and', 89722),
     ('a', 83688),
     ('of', 76855),
     ('to', 66746),
     ('is', 57245),
     ('in', 50215),
     ('br', 49235),
     ('it', 48025),
     ('i', 40743),
     ('that', 35630),
     ('this', 35080),
     ('s', 33815),
     ('as', 26308),
     ('with', 23247),
     ('for', 22416),
     ('was', 21917),
     ('film', 20937),
     ('but', 20822),
     ('movie', 19074),
     ('his', 17227),
     ('on', 17008),
     ('you', 16681),
     ('he', 16282),
     ('are', 14807),
     ('not', 14272),
     ('t', 13720),
     ('one', 13655),
     ('have', 12587),
     ('be', 12416),
     ('by', 11997),
     ('all', 11942),
     ('who', 11464),
     ('an', 11294),
     ('at', 11234),
     ('from', 10767),
     ('her', 10474),
     ('they', 9895),
     ('has', 9186),
     ('so', 9154),
     ('like', 9038),
     ('about', 8313),
     ('very', 8305),
     ('out', 8134),
     ('there', 8057),
     ('she', 7779),
     ('what', 7737),
     ('or', 7732)]



This has produces a collection of the positive and negarive words and their number of occurances. Scrolling through the data clearly shows that there is a difference between positive and negative word occurances. However straight away we can see the data is cluttered with useless words. 

A good step to resolve this is to find the ratio between the words occurance in the positive set and the negative set.


```python
pos_neg_ratios = Counter()

for term, cnt in list(total_counts.most_common()):
    if cnt > 100:
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
        pos_neg_ratios[term] = pos_neg_ratio
        
for word, ratio in pos_neg_ratios.most_common():
    if ratio > 1:
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))
```

Loop through each word in the total count where term is the word and cnt is the number of occurances. If the number of occurances is greater than 100 calculate the ratio between the occurances of that word in the positive list and the negative list and add it to the counter.

Then, loop through all the new ratios where word is the word and ratio is the ratio calculated in the last step. If the ratio is greater than 1 it will occur more in the positive set than the negative set. Calculate the natural log to normalise. Otherwise calculate 1 / the natrural log and make it negative. the 0.01 is added so there are no divde by 0 errors.


```python
# Words most frequently seen in the postive reviews
pos_neg_ratios.most_common()[0:50]
```




    [('edie', 4.6913478822291435),
     ('paulie', 4.0775374439057197),
     ('felix', 3.1527360223636558),
     ('polanski', 2.8233610476132043),
     ('matthau', 2.8067217286092401),
     ('victoria', 2.6810215287142909),
     ('mildred', 2.6026896854443837),
     ('gandhi', 2.5389738710582761),
     ('flawless', 2.451005098112319),
     ('superbly', 2.2600254785752498),
     ('perfection', 2.1594842493533721),
     ('astaire', 2.1400661634962708),
     ('captures', 2.0386195471595809),
     ('voight', 2.0301704926730531),
     ('wonderfully', 2.0218960560332353),
     ('powell', 1.9783454248084671),
     ('brosnan', 1.9547990964725592),
     ('lily', 1.9203768470501485),
     ('bakshi', 1.9029851043382795),
     ('lincoln', 1.9014583864844796),
     ('refreshing', 1.8551812956655511),
     ('breathtaking', 1.8481124057791867),
     ('bourne', 1.8478489358790986),
     ('lemmon', 1.8458266904983307),
     ('delightful', 1.8002701588959635),
     ('flynn', 1.7996646487351682),
     ('andrews', 1.7764919970972666),
     ('homer', 1.7692866133759964),
     ('beautifully', 1.7626953362841438),
     ('soccer', 1.7578579175523736),
     ('elvira', 1.7397031072720019),
     ('underrated', 1.7197859696029656),
     ('gripping', 1.7165360479904674),
     ('superb', 1.7091514458966952),
     ('delight', 1.6714733033535532),
     ('welles', 1.6677068205580761),
     ('sadness', 1.663505133704376),
     ('sinatra', 1.6389967146756448),
     ('touching', 1.637217476541176),
     ('timeless', 1.62924053973028),
     ('macy', 1.6211339521972916),
     ('unforgettable', 1.6177367152487956),
     ('favorites', 1.6158688027643908),
     ('stewart', 1.6119987332957739),
     ('sullivan', 1.6094379124341003),
     ('extraordinary', 1.6094379124341003),
     ('hartley', 1.6094379124341003),
     ('brilliantly', 1.5950491749820008),
     ('friendship', 1.5677652160335325),
     ('wonderful', 1.5645425925262093)]




```python
# Words most frequently seen in the negative reviews
list(reversed(pos_neg_ratios.most_common()))[0:30]
```




    [('boll', -4.0778152602708904),
     ('uwe', -3.9218753018711578),
     ('seagal', -3.3202501058581921),
     ('unwatchable', -3.0269848170580955),
     ('stinker', -2.9876839403711624),
     ('mst', -2.7753833211707968),
     ('incoherent', -2.7641396677532537),
     ('unfunny', -2.5545257844967644),
     ('waste', -2.4907515123361046),
     ('blah', -2.4475792789485005),
     ('horrid', -2.3715779644809971),
     ('pointless', -2.3451073877136341),
     ('atrocious', -2.3187369339642556),
     ('redeeming', -2.2667790015910296),
     ('prom', -2.2601040980178784),
     ('drivel', -2.2476029585766928),
     ('lousy', -2.2118080125207054),
     ('worst', -2.1930856334332267),
     ('laughable', -2.172468615469592),
     ('awful', -2.1385076866397488),
     ('poorly', -2.1326133844207011),
     ('wasting', -2.1178155545614512),
     ('remotely', -2.111046881095167),
     ('existent', -2.0024805005437076),
     ('boredom', -1.9241486572738005),
     ('miserably', -1.9216610938019989),
     ('sucks', -1.9166645809588516),
     ('uninspired', -1.9131499212248517),
     ('lame', -1.9117232884159072),
     ('insult', -1.9085323769376259)]



## Transforming text into numbers

The current information is usefull to those of us who can already read but not very usefull to a neural network!
Before we train the neural network we must convert the data into a format that can be used for in the network!


```python
vocab = set(total_counts.keys())
vocab_size = len(vocab)
print ('The number of different words in the data: ', vocab_size)
```

    The number of different words in the data:  74074



```python
list(vocab)[0:50]
```




    ['',
     'believable',
     'precedes',
     'cineliterate',
     'jubilee',
     'seediness',
     'grinchy',
     'neeson',
     'kaleidoscope',
     'jogando',
     'fractured',
     'krebs',
     'villedo',
     'sighting',
     'cleavers',
     'ivanova',
     'compositions',
     'checkpoints',
     'consummated',
     'yore',
     'penciled',
     'hitlerian',
     'cybil',
     'frflutet',
     'roseanne',
     'evos',
     'corral',
     'khushi',
     'farily',
     'hairstyle',
     'dreamily',
     'hoots',
     'miles',
     'andersen',
     'coverups',
     'seraphic',
     'listenable',
     'maywether',
     'thompson',
     'byways',
     'congratulates',
     'probe',
     'outwardly',
     'heats',
     'jaliyl',
     'nickels',
     'visionaries',
     'splinters',
     'wienberg',
     'arsenical']



vocab is a list of all the different words found in the dataset

### Creating the input and output data


```python
import numpy as np

layer_0 = np.zeros((1, vocab_size))
layer_0
```




    array([[ 0.,  0.,  0., ...,  0.,  0.,  0.]])



First we create the input layer with a length of the number of different words. We fill it with zeros to save memory which can improve comutiational efficiency.

Next, take each word and give it a unique index:


```python
word2index = {}

for i, word in enumerate(vocab):
    word2index[word] = i

word2index
```




    {'': 0,
     'believable': 1,
     'precedes': 2,
     'cineliterate': 3,
     'jubilee': 4,
     'seediness': 5,
     'grinchy': 6,
     'neeson': 7,
     'kaleidoscope': 8,
     'jogando': 9,
     'fractured': 10,
     'krebs': 11,
     'villedo': 12,
     'sighting': 13,
     'cleavers': 14,
     'ivanova': 15,
     'compositions': 16,
     'checkpoints': 17,
     'consummated': 18,
     'yore': 19,
     'penciled': 20,
     'hitlerian': 21,
     'cybil': 22,
     'frflutet': 23,
     'roseanne': 24,
     'evos': 25,
     'corral': 26,
     'khushi': 27,
     'farily': 28,
     'hairstyle': 29,
     'dreamily': 30,
     'hoots': 31,
     'miles': 32,
     'andersen': 33,
     'coverups': 34,
     'seraphic': 35,
     'listenable': 36,
     'maywether': 37,
     'thompson': 38,
     'byways': 39,
     'congratulates': 40,
     'probe': 41,
     'outwardly': 42,
     'heats': 43,
     'jaliyl': 44,
     'nickels': 45,
     'visionaries': 46,
     'splinters': 47,
     'wienberg': 48,
     'arsenical': 49,
     'campbell': 50,
     'nutritional': 51,
     'eravamo': 52,
     'zasu': 53,
     'beasties': 54,
     'junk': 55,
     'registrar': 56,
     'milder': 57,
     'shortland': 58,
     'squatting': 59,
     'gossamer': 60,
     'practiced': 61,
     'retentiveness': 62,
     'dattilo': 63,
     'telefilm': 64,
     'unpleasantly': 65,
     'freshman': 66,
     'scob': 67,
     'masssacre': 68,
     'auds': 69,
     'restructured': 70,
     'eightstars': 71,
     'souring': 72,
     'forgery': 73,
     'cheezily': 74,
     'pleasures': 75,
     'sisyphus': 76,
     'conserving': 77,
     'micheaux': 78,
     'khallas': 79,
     'dogie': 80,
     'crappiest': 81,
     'inhibitions': 82,
     'dachau': 83,
     'pastry': 84,
     'particulars': 85,
     'childishly': 86,
     'piemaker': 87,
     'garbage': 88,
     'jewelery': 89,
     'sipus': 90,
     'recurrent': 91,
     'barter': 92,
     'yeun': 93,
     'institutionalization': 94,
     'intrinsically': 95,
     'speckle': 96,
     'adelade': 97,
     'inheriting': 98,
     'boarding': 99,
     'loaded': 100,
     'flutes': 101,
     'deadhead': 102,
     'pummeled': 103,
     'thenprepare': 104,
     'comparisons': 105,
     'radford': 106,
     'fairfaix': 107,
     'littlesearch': 108,
     'ismael': 109,
     'mitzi': 110,
     'elways': 111,
     'panoply': 112,
     'quade': 113,
     'adhere': 114,
     'euthanized': 115,
     'melyvn': 116,
     'stereotyped': 117,
     'vastly': 118,
     'census': 119,
     'piled': 120,
     'lili': 121,
     'gamboa': 122,
     'mwah': 123,
     'nui': 124,
     'climatic': 125,
     'trickery': 126,
     'imperative': 127,
     'soapbox': 128,
     'ransacking': 129,
     'bluster': 130,
     'jerked': 131,
     'ifyou': 132,
     'connived': 133,
     'thickening': 134,
     'gorbunov': 135,
     'riccardo': 136,
     'unidimensional': 137,
     'multilevel': 138,
     'especically': 139,
     'dailys': 140,
     'stairway': 141,
     'instructors': 142,
     'unicycle': 143,
     'snozzcumbers': 144,
     'exclaimed': 145,
     'cambpell': 146,
     'encouraged': 147,
     'width': 148,
     'comas': 149,
     'leontine': 150,
     'conelley': 151,
     'oom': 152,
     'mazurki': 153,
     'exelence': 154,
     'messick': 155,
     'xer': 156,
     'peer': 157,
     'lalouche': 158,
     'harmonized': 159,
     'lagosi': 160,
     'unbecomingly': 161,
     'march': 162,
     'lodge': 163,
     'toughie': 164,
     'workshop': 165,
     'educate': 166,
     'hairs': 167,
     'bregovic': 168,
     'entice': 169,
     'woulda': 170,
     'beffe': 171,
     'bathouse': 172,
     'yahoo': 173,
     'workman': 174,
     'padre': 175,
     'pricks': 176,
     'phesht': 177,
     'metropolis': 178,
     'nhk': 179,
     'chattarjee': 180,
     'stumbled': 181,
     'spruce': 182,
     'anachronic': 183,
     'mauling': 184,
     'grimms': 185,
     'mimetic': 186,
     'clastrophobic': 187,
     'antiques': 188,
     'trepidous': 189,
     'amrohi': 190,
     'clenches': 191,
     'acing': 192,
     'comptent': 193,
     'wavered': 194,
     'hawki': 195,
     'luchino': 196,
     'gutters': 197,
     'rinsing': 198,
     'elle': 199,
     'misinformation': 200,
     'alejo': 201,
     'fille': 202,
     'plumb': 203,
     'jogger': 204,
     'devalues': 205,
     'thank': 206,
     'lorded': 207,
     'presents': 208,
     'eights': 209,
     'wan': 210,
     'overreacting': 211,
     'stinkers': 212,
     'acd': 213,
     'tragic': 214,
     'grimaces': 215,
     'sutra': 216,
     'patio': 217,
     'blossomed': 218,
     'caro': 219,
     'warlords': 220,
     'isthar': 221,
     'enjoyably': 222,
     'porsche': 223,
     'healey': 224,
     'daniela': 225,
     'agonia': 226,
     'involution': 227,
     'strafe': 228,
     'olivia': 229,
     'tenets': 230,
     'authorty': 231,
     'prettiest': 232,
     'shoals': 233,
     'hussy': 234,
     'reiser': 235,
     'luxembourg': 236,
     'pickers': 237,
     'migratory': 238,
     'airlessness': 239,
     'fashionthat': 240,
     'emote': 241,
     'rhetorics': 242,
     'aspirations': 243,
     'pojar': 244,
     'fiendish': 245,
     'dungy': 246,
     'anywhoo': 247,
     'floater': 248,
     'vulgarly': 249,
     'eddi': 250,
     'documentarian': 251,
     'gratingly': 252,
     'latinity': 253,
     'saurious': 254,
     'nearby': 255,
     'rickety': 256,
     'cinema': 257,
     'amateurs': 258,
     'molina': 259,
     'cartmans': 260,
     'renn': 261,
     'gangbangers': 262,
     'bulked': 263,
     'zungia': 264,
     'remarry': 265,
     'unison': 266,
     'raking': 267,
     'stepfather': 268,
     'assign': 269,
     'rove': 270,
     'garlands': 271,
     'klaws': 272,
     'venturing': 273,
     'escalates': 274,
     'gwyne': 275,
     'commands': 276,
     'portal': 277,
     'killshot': 278,
     'coolio': 279,
     'isham': 280,
     'peahi': 281,
     'nicola': 282,
     'producing': 283,
     'baja': 284,
     'picket': 285,
     'egging': 286,
     'sterno': 287,
     'chetniks': 288,
     'yo': 289,
     'minutia': 290,
     'p': 291,
     'pink': 292,
     'sawahla': 293,
     'janice': 294,
     'bondy': 295,
     'bastidge': 296,
     'expire': 297,
     'newscaster': 298,
     'crewmate': 299,
     'physician': 300,
     'unintentional': 301,
     'philp': 302,
     'gingold': 303,
     'cupertino': 304,
     'koersk': 305,
     'obituaries': 306,
     'carmilla': 307,
     'sphincters': 308,
     'attainable': 309,
     'sly': 310,
     'ceaseless': 311,
     'omnibus': 312,
     'harline': 313,
     'cheddar': 314,
     'horribleness': 315,
     'lezlie': 316,
     'succinctly': 317,
     'renounce': 318,
     'yr': 319,
     'sugimoto': 320,
     'bracketed': 321,
     'alexanader': 322,
     'nineteen': 323,
     'roofer': 324,
     'bocabonita': 325,
     'increses': 326,
     'wotw': 327,
     'daft': 328,
     'autumn': 329,
     'proverbs': 330,
     'akelly': 331,
     'corresponding': 332,
     'ellens': 333,
     'rattles': 334,
     'dateless': 335,
     'lancaster': 336,
     'ht': 337,
     'tobacconist': 338,
     'ow': 339,
     'milliagn': 340,
     'intersected': 341,
     'mound': 342,
     'fantasising': 343,
     'prescient': 344,
     'bahgdad': 345,
     'mangini': 346,
     'anthropophagous': 347,
     'structural': 348,
     'hill': 349,
     'habit': 350,
     'schaeffer': 351,
     'mailroom': 352,
     'unknowingly': 353,
     'greedo': 354,
     'fascinating': 355,
     'gainsbrough': 356,
     'wars': 357,
     'unfunnily': 358,
     'apologies': 359,
     'sauvage': 360,
     'junebug': 361,
     'registration': 362,
     'beavers': 363,
     'rei': 364,
     'paraphrases': 365,
     'courrieres': 366,
     'dada': 367,
     'andress': 368,
     'uninformative': 369,
     'plaudits': 370,
     'bertie': 371,
     'pynchon': 372,
     'abolition': 373,
     'preponderance': 374,
     'congorilla': 375,
     'cateress': 376,
     'harilal': 377,
     'piso': 378,
     'quien': 379,
     'disobeying': 380,
     'deamon': 381,
     'flannery': 382,
     'dedicated': 383,
     'merged': 384,
     'miyoshi': 385,
     'form': 386,
     'picked': 387,
     'stubborn': 388,
     'expressiveness': 389,
     'olivier': 390,
     'abstract': 391,
     'cambridge': 392,
     'cheesiest': 393,
     'coleen': 394,
     'prevents': 395,
     'josten': 396,
     'psychomania': 397,
     'enviably': 398,
     'unspoken': 399,
     'foibles': 400,
     'illustriousness': 401,
     'thesinger': 402,
     'nea': 403,
     'baloo': 404,
     'stakeout': 405,
     'ahhhhhh': 406,
     'prehensile': 407,
     'dwindle': 408,
     'subsides': 409,
     'nighty': 410,
     'visualizes': 411,
     'harass': 412,
     'inventions': 413,
     'compactor': 414,
     'quibbles': 415,
     'tantric': 416,
     'preetam': 417,
     'bushi': 418,
     'splendid': 419,
     'pallance': 420,
     'videodisc': 421,
     'zardkuh': 422,
     'leaped': 423,
     'choses': 424,
     'kolos': 425,
     'reintegration': 426,
     'hatsumomo': 427,
     'bourn': 428,
     'touches': 429,
     'barek': 430,
     'specific': 431,
     'mounties': 432,
     'yay': 433,
     'candlelit': 434,
     'chulawasse': 435,
     'provisional': 436,
     'allegedly': 437,
     'attach': 438,
     'sensors': 439,
     'distended': 440,
     'talledega': 441,
     'maccarthy': 442,
     'kicking': 443,
     'marm': 444,
     'predeccesor': 445,
     'minneli': 446,
     'barraged': 447,
     'vitametavegamin': 448,
     'spacesuit': 449,
     'minister': 450,
     'ashura': 451,
     'aghast': 452,
     'utilize': 453,
     'pities': 454,
     'unaccomplished': 455,
     'jyaada': 456,
     'inexperience': 457,
     'candlestick': 458,
     'inconclusive': 459,
     'published': 460,
     'groot': 461,
     'overprotective': 462,
     'enthuses': 463,
     'brownish': 464,
     'bds': 465,
     'chrissakes': 466,
     'minded': 467,
     'fangs': 468,
     'tiefenbach': 469,
     'nang': 470,
     'hercule': 471,
     'hofsttter': 472,
     'mongrel': 473,
     'islands': 474,
     'throwaway': 475,
     'calamai': 476,
     'warnicki': 477,
     'lied': 478,
     'dismalness': 479,
     'transplanting': 480,
     'jarryd': 481,
     'commit': 482,
     'cora': 483,
     'aryeman': 484,
     'lowlife': 485,
     'sunnybrook': 486,
     'christan': 487,
     'jest': 488,
     'observed': 489,
     'chronicled': 490,
     'monstervision': 491,
     'obsessed': 492,
     'meanest': 493,
     'timetable': 494,
     'whereby': 495,
     'suffers': 496,
     'tomei': 497,
     'mix': 498,
     'submerges': 499,
     'ganzel': 500,
     'shekhar': 501,
     'relocations': 502,
     'perpetual': 503,
     'absconded': 504,
     'fountained': 505,
     'occident': 506,
     'uncountable': 507,
     'huntingdon': 508,
     'jeering': 509,
     'katell': 510,
     'szubanski': 511,
     'speirs': 512,
     'investors': 513,
     'whys': 514,
     'graceful': 515,
     'cinematograph': 516,
     'destructible': 517,
     'scrat': 518,
     'skeweredness': 519,
     'crudest': 520,
     'bismol': 521,
     'perceived': 522,
     'sheryll': 523,
     'grift': 524,
     'medencevic': 525,
     'wearisome': 526,
     'epigram': 527,
     'transliteration': 528,
     'liken': 529,
     'fads': 530,
     'trivialise': 531,
     'guardians': 532,
     'nav': 533,
     'scates': 534,
     'for': 535,
     'migrated': 536,
     'gentlemanlike': 537,
     'dived': 538,
     'hainey': 539,
     'cashew': 540,
     'pretzels': 541,
     'pakistani': 542,
     'organzation': 543,
     'easting': 544,
     'pseudonyms': 545,
     'philosophers': 546,
     'ghost': 547,
     'kostas': 548,
     'moto': 549,
     'mujhe': 550,
     'striba': 551,
     'gentle': 552,
     'enslaved': 553,
     'indictment': 554,
     'philadelphia': 555,
     'forgiving': 556,
     'levitating': 557,
     'fumes': 558,
     'luz': 559,
     'persuade': 560,
     'conrow': 561,
     'surrealistic': 562,
     'wrested': 563,
     'subcommander': 564,
     'notrious': 565,
     'trojans': 566,
     'comming': 567,
     'jumping': 568,
     'disabled': 569,
     'homeboys': 570,
     'rawail': 571,
     'tykwer': 572,
     'walterman': 573,
     'meself': 574,
     'mushed': 575,
     'failure': 576,
     'empahh': 577,
     'dionna': 578,
     'continental': 579,
     'loudest': 580,
     'duel': 581,
     'terrorvision': 582,
     'lorain': 583,
     'chronicle': 584,
     'grusiya': 585,
     'dullness': 586,
     'hitcock': 587,
     'steinem': 588,
     'takeout': 589,
     'have': 590,
     'balaji': 591,
     'sausages': 592,
     'ikwydls': 593,
     'aren': 594,
     'faludi': 595,
     'postmortem': 596,
     'fantasizing': 597,
     'izes': 598,
     'vikki': 599,
     'grandparent': 600,
     'snowed': 601,
     'fixx': 602,
     'outcomes': 603,
     'kazaam': 604,
     'insufficiency': 605,
     'rydell': 606,
     'frat': 607,
     'substantive': 608,
     'spinoff': 609,
     'unbroken': 610,
     'valve': 611,
     'westbridbe': 612,
     'nickelodeon': 613,
     'mida': 614,
     'nuns': 615,
     'quoted': 616,
     'arzner': 617,
     'hagiography': 618,
     'widower': 619,
     'nourish': 620,
     'cheezoid': 621,
     'returns': 622,
     'outfielder': 623,
     'bushes': 624,
     'fluctuations': 625,
     'casnoff': 626,
     'conflict': 627,
     'diagnosed': 628,
     'klebb': 629,
     'elseavoid': 630,
     'marching': 631,
     'sacco': 632,
     'smeaton': 633,
     'louella': 634,
     'gwynne': 635,
     'reddish': 636,
     'delauise': 637,
     'flixmedia': 638,
     'ratcatcher': 639,
     'unengineered': 640,
     'masturbation': 641,
     'strenuous': 642,
     'guayabera': 643,
     'clubgoer': 644,
     'wegener': 645,
     'nanosecond': 646,
     'aicha': 647,
     'cradle': 648,
     'imoogi': 649,
     'childishness': 650,
     'talks': 651,
     'intransigent': 652,
     'anticipatory': 653,
     'skepticism': 654,
     'gall': 655,
     'ithaca': 656,
     'josie': 657,
     'cortese': 658,
     'storymode': 659,
     'gurney': 660,
     'beheaded': 661,
     'concurrently': 662,
     'coupon': 663,
     'traffics': 664,
     'nightstalker': 665,
     'midori': 666,
     'iwaya': 667,
     'mundanity': 668,
     'monitored': 669,
     'clearlly': 670,
     'between': 671,
     'vessela': 672,
     'ilu': 673,
     'fibbed': 674,
     'tenniel': 675,
     'stuffiness': 676,
     'clint': 677,
     'smartly': 678,
     'unnamed': 679,
     'peak': 680,
     'giff': 681,
     'longhair': 682,
     'aman': 683,
     'disgruntle': 684,
     'wise': 685,
     'squirted': 686,
     'slaving': 687,
     'muoz': 688,
     'wrestlers': 689,
     'sloane': 690,
     'rawal': 691,
     'voiceovers': 692,
     'knarl': 693,
     'paramedic': 694,
     'borlenghi': 695,
     'helge': 696,
     'linney': 697,
     'czarist': 698,
     'kafka': 699,
     'infesting': 700,
     'wath': 701,
     'transcendant': 702,
     'goobacks': 703,
     'doyeon': 704,
     'bourgeois': 705,
     'roget': 706,
     'makepease': 707,
     'lancie': 708,
     'wellesley': 709,
     'refusals': 710,
     'condoning': 711,
     'dudek': 712,
     'bergerac': 713,
     'sala': 714,
     'shaka': 715,
     'yester': 716,
     'yumiko': 717,
     'cooking': 718,
     'dion': 719,
     'benjiman': 720,
     'packy': 721,
     'praying': 722,
     'outdid': 723,
     'known': 724,
     'goldenhersh': 725,
     'replicas': 726,
     'rafi': 727,
     'gotb': 728,
     'swathes': 729,
     'rages': 730,
     'emotional': 731,
     'napton': 732,
     'aligned': 733,
     'squeel': 734,
     'stimulating': 735,
     'dearing': 736,
     'cancer': 737,
     'kansasi': 738,
     'luby': 739,
     'ste': 740,
     'kidd': 741,
     'palaces': 742,
     'procreate': 743,
     'telegraphed': 744,
     'unify': 745,
     'christers': 746,
     'impaled': 747,
     'babaloo': 748,
     'flav': 749,
     'imbued': 750,
     'towed': 751,
     'dumb': 752,
     'heartaches': 753,
     'equipped': 754,
     'wyne': 755,
     'eardrum': 756,
     'nietzche': 757,
     'call': 758,
     'expects': 759,
     'bladerunner': 760,
     'snarl': 761,
     'burnside': 762,
     'likening': 763,
     'intertextuality': 764,
     'dreamquest': 765,
     'bigelow': 766,
     'puritan': 767,
     'creds': 768,
     'scoffs': 769,
     'brusquely': 770,
     'readjusts': 771,
     'imagined': 772,
     'involves': 773,
     'emerge': 774,
     'tapestries': 775,
     'tubbs': 776,
     'mysoju': 777,
     'donation': 778,
     'spokane': 779,
     'indirection': 780,
     'championships': 781,
     'schotland': 782,
     'sussanah': 783,
     'soccer': 784,
     'tagged': 785,
     'greenwood': 786,
     'travails': 787,
     'separation': 788,
     'pointless': 789,
     'blazer': 790,
     'moviehowever': 791,
     'roaring': 792,
     'curt': 793,
     'billingsley': 794,
     'withnail': 795,
     'meatloaf': 796,
     'pillman': 797,
     'ching': 798,
     'philedelphia': 799,
     'nausicaa': 800,
     'shipments': 801,
     'neha': 802,
     'nesher': 803,
     'goldmember': 804,
     'poet': 805,
     'shockumenary': 806,
     'cinderellas': 807,
     'madnes': 808,
     'resolve': 809,
     'enhances': 810,
     'stroll': 811,
     'jogs': 812,
     'ger': 813,
     'liebman': 814,
     'htm': 815,
     'aquart': 816,
     'freq': 817,
     'foreveror': 818,
     'circulating': 819,
     'bapu': 820,
     'soupy': 821,
     'caesars': 822,
     'trelkovski': 823,
     'definative': 824,
     'caffeine': 825,
     'alloted': 826,
     'writter': 827,
     'stead': 828,
     'daffodils': 829,
     'mineshaft': 830,
     'sugary': 831,
     'invocus': 832,
     'baguette': 833,
     'mongkok': 834,
     'vindicated': 835,
     'consolidate': 836,
     'ognianova': 837,
     'tuff': 838,
     'thema': 839,
     'hulkamaniacs': 840,
     'catlike': 841,
     'whoppie': 842,
     'tanga': 843,
     'plagues': 844,
     'purply': 845,
     'lear': 846,
     'accidently': 847,
     'wrist': 848,
     'instigators': 849,
     'barroom': 850,
     'ohana': 851,
     'setups': 852,
     'wad': 853,
     'gaolers': 854,
     'doorstop': 855,
     'darwininan': 856,
     'wwaste': 857,
     'doesn': 858,
     'despots': 859,
     'appreciating': 860,
     'thieson': 861,
     'savor': 862,
     'stupifyingly': 863,
     'rollerblades': 864,
     'stuttured': 865,
     'artsieness': 866,
     'scanning': 867,
     'albas': 868,
     'lodger': 869,
     'mussolini': 870,
     'saboteur': 871,
     'stammering': 872,
     'caries': 873,
     'dadaist': 874,
     'montes': 875,
     'troi': 876,
     'chooser': 877,
     'creditable': 878,
     'loutish': 879,
     'potentialities': 880,
     'logics': 881,
     'differentiates': 882,
     'ancestral': 883,
     'rack': 884,
     'grrr': 885,
     'squawk': 886,
     'ugo': 887,
     'captains': 888,
     'moor': 889,
     'chador': 890,
     'guise': 891,
     'corpse': 892,
     'deviates': 893,
     'uav': 894,
     'gloomily': 895,
     'valga': 896,
     'marguis': 897,
     'valentinov': 898,
     'berseker': 899,
     'truer': 900,
     'kidnapping': 901,
     'lamebrained': 902,
     'conjoined': 903,
     'premade': 904,
     'viewed': 905,
     'steels': 906,
     'hackery': 907,
     'edina': 908,
     'maitlan': 909,
     'solino': 910,
     'krummernes': 911,
     'annick': 912,
     'skinned': 913,
     'noriaki': 914,
     'berrymore': 915,
     'tainos': 916,
     'gourmet': 917,
     'constricted': 918,
     'natassja': 919,
     'hsd': 920,
     'hieroglyphs': 921,
     'omit': 922,
     'righted': 923,
     'perished': 924,
     'dza': 925,
     'cider': 926,
     'sailfish': 927,
     'peaces': 928,
     'corporate': 929,
     'clicks': 930,
     'bungalow': 931,
     'flaps': 932,
     'doughty': 933,
     'bastions': 934,
     'appaloosa': 935,
     'unvarnished': 936,
     'unsupportive': 937,
     'dependants': 938,
     'enigmas': 939,
     'caster': 940,
     'chicas': 941,
     'deteriorate': 942,
     'woth': 943,
     'bufford': 944,
     'ato': 945,
     'nat': 946,
     'showthe': 947,
     'doubtfire': 948,
     'tdd': 949,
     'deneuve': 950,
     'crosses': 951,
     'threatened': 952,
     'insulate': 953,
     'zippier': 954,
     'faithfully': 955,
     'forgotten': 956,
     'faggot': 957,
     'argue': 958,
     'vulgarism': 959,
     'foppish': 960,
     'hallelujah': 961,
     'ortiz': 962,
     'disarray': 963,
     'impersonator': 964,
     'tipping': 965,
     'jerichow': 966,
     'hallie': 967,
     'kovacs': 968,
     'hough': 969,
     'mundanely': 970,
     'andrenaline': 971,
     'idaho': 972,
     'hubba': 973,
     'deadset': 974,
     'vampyr': 975,
     'verbaan': 976,
     'luchi': 977,
     'taratino': 978,
     'suffice': 979,
     'demin': 980,
     'employes': 981,
     'vassilis': 982,
     'cardenas': 983,
     'visage': 984,
     'moviesone': 985,
     'toyland': 986,
     'plagiary': 987,
     'underused': 988,
     'espoused': 989,
     'aloud': 990,
     'resized': 991,
     'ghunghroo': 992,
     'outline': 993,
     'stain': 994,
     'carfare': 995,
     'crucial': 996,
     'expense': 997,
     'quieted': 998,
     'breakups': 999,
     ...}



Next we need to update the input layer to actually contain the information and not just zeros.


```python
def update_input_layer(review):
    
    global layer_0
    
    # Ensure all of layer 0 is set to 0
    layer_0 *= 0
    
    for word in review.split(' '):
        layer_0[0][word2index[word]] += 1
        
update_input_layer(reviews[0])
```

This method takes a review and loops through each word. It then adds 1 to the value at the index of the word2index number of the word. This will build a collection of the number of word occurances at the unique index of each word.


```python
layer_0
```




    array([[ 18.,   0.,   0., ...,   0.,   0.,   0.]])



Finally, we need to get the 'POSITIVE' and 'NEGATIVE' labels into a machine readable format. Here we will use a 1 or 0


```python
def get_target_for_label(label):
    if label == 'POSITIVE':
        return 1
    else:
        return 0
```


```python
labels[0]
```




    'POSITIVE'




```python
get_target_for_label(labels[0])
```




    1



## Building the network

For this I am going to use the NeuralNetwork class from Udacity Project 1 with some adjustments:
- 3 Layer neural network
- no non-linearity in the second layer (No sigmoid between layer 0 and 1)
- use previous functions to create the training data
- create a 'pre_process_data' function to create vocabulary for the training data and generating functions
- modify train to train over the entire corpus


```python
import numpy as np
import time
import sys

class SentimentNetwork():
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        
        # Seed the random number generator for debugging
        np.random.seed(1)
        
        self.pre_process_data(reviews, labels)
        
        self.init_network(self.review_vocab_size, hidden_nodes, 1, learning_rate)
    
    # Process all the review and label data and form unique dictionarys
    # Give each word a unique index for entry into the network
    def pre_process_data(self, reviews, labels):
        # Creates a dictionary that contains one of each different word
        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):
                review_vocab.add(word)
        
        self.review_vocab = list(review_vocab)
        
        # Creates a dictionary that contains one of each label type [IN our case 'POSITIVE' or 'NEGATIVE']
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
                
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of each dictionary
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Give each word a uniquely identifying index
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        # Give each label an identifying index [in our case only 0 or 1]
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
    
    # Initialize all the network parameters
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes for all three layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # Initialise weights
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1, input_nodes))
    
    # Activation function
    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))
    
    # Reset the layer for the next pass and add the new words from the review
    def update_input_layer(self, review):
        # Clear the previous state and set to 0
        self.layer_0 *= 0
        
        for word in review.split(' '):
            if (word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] += 1
    
    # Get the neumerical reprisentation of the output
    def get_target_for_label(self, label):
        if label == 'POSITIVE':
            return 1
        else:
            return 0
    
    # Train the network
    def train(self, training_reviews, training_labels):
        
        # Check that the inputs match the outputs
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        # Log the start time
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            ## Forward Pass ##
            
            # Input Layer
            self.update_input_layer(review)
            
            # Hidden Layer
            layer_1 = self.layer_0.dot(self.weights_0_1)
            
            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
            
            ## Backwards Pass ##
            
            # Output Error
            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid(layer_2, True)
            
            # Hidden Error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error
            
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate
            
            if (np.abs(layer_2_error) < 0.5):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    # Test the network
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            
            if pred == testing_labels[i]:
                correct += 1
                
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                            + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    # Forward propogate        
    def run(self, review):
        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
```

Next we need to create an instance of the network and train it on the data.


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
```


```python
# evaluate our model before training (just to show how horrible it is)
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):915.9% #Correct:500 #Tested:1000 Testing Accuracy:50.0%


```python
# train the network
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:0.0% Speed(reviews/sec):0.0 #Correct:0 #Trained:1 Training Accuracy:0.0%
    Progress:10.4% Speed(reviews/sec):177.1 #Correct:1250 #Trained:2501 Training Accuracy:49.9%
    Progress:20.8% Speed(reviews/sec):173.9 #Correct:2500 #Trained:5001 Training Accuracy:49.9%
    Progress:31.2% Speed(reviews/sec):179.8 #Correct:3750 #Trained:7501 Training Accuracy:49.9%
    Progress:41.6% Speed(reviews/sec):182.9 #Correct:5000 #Trained:10001 Training Accuracy:49.9%
    Progress:52.0% Speed(reviews/sec):183.7 #Correct:6250 #Trained:12501 Training Accuracy:49.9%
    Progress:62.5% Speed(reviews/sec):179.2 #Correct:7500 #Trained:15001 Training Accuracy:49.9%
    Progress:72.9% Speed(reviews/sec):181.3 #Correct:8750 #Trained:17501 Training Accuracy:49.9%
    Progress:83.3% Speed(reviews/sec):183.0 #Correct:10000 #Trained:20001 Training Accuracy:49.9%
    Progress:93.7% Speed(reviews/sec):185.2 #Correct:11250 #Trained:22501 Training Accuracy:49.9%
    Progress:99.9% Speed(reviews/sec):186.3 #Correct:11999 #Trained:24000 Training Accuracy:49.9%

The result of training was actually worse than predicting the outcome randomly. Adjustments to the learning rate had little affect and therefore there must be something wrong with the data.

## Neural Noise

Neural noise is when the important data is being drowned out by other pieces of unimportant data. 

An analagy would be:
>the neural network is a spade and it helps you dig for gold. However no type of spade is going to help you get more gold if you are digging in the wrong place.

To solve this we need to eliminate the noise in the data. First, lets check the data to see if there is anything that shouldn't be there


```python
# The function defined previously
def update_input_layer(review):
    
    global layer_0
    
    # clear out previous state, reset the layer to be all 0s
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

update_input_layer(reviews[0])
```


```python
layer_0
```




    array([[ 18.,   0.,   0., ...,   0.,   0.,   0.]])




```python
review_counter = Counter()

for word in reviews[0].split(' '):
    review_counter[word] += 1
    
review_counter.most_common()[0:50]
```




    [('.', 27),
     ('', 18),
     ('the', 9),
     ('to', 6),
     ('high', 5),
     ('i', 5),
     ('bromwell', 4),
     ('is', 4),
     ('a', 4),
     ('teachers', 4),
     ('that', 4),
     ('of', 4),
     ('it', 2),
     ('at', 2),
     ('as', 2),
     ('school', 2),
     ('my', 2),
     ('in', 2),
     ('me', 2),
     ('students', 2),
     ('their', 2),
     ('student', 2),
     ('cartoon', 1),
     ('comedy', 1),
     ('ran', 1),
     ('same', 1),
     ('time', 1),
     ('some', 1),
     ('other', 1),
     ('programs', 1),
     ('about', 1),
     ('life', 1),
     ('such', 1),
     ('years', 1),
     ('teaching', 1),
     ('profession', 1),
     ('lead', 1),
     ('believe', 1),
     ('s', 1),
     ('satire', 1),
     ('much', 1),
     ('closer', 1),
     ('reality', 1),
     ('than', 1),
     ('scramble', 1),
     ('survive', 1),
     ('financially', 1),
     ('insightful', 1),
     ('who', 1),
     ('can', 1)]



Looking at the data we can see there are 18 instances of an empty space. As this does not convey the sentiment in any way it must be removed.

## Reducing noise in the input data

In this situation we can see that the data is being scewed by the large number of useless values. We can resolve this by changing the update_input_layer method to not incriment the word count but just log it as present!


```python
import numpy as np
import time
import sys

class SentimentNetwork():
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        
        # Seed the random number generator for debugging
        np.random.seed(1)
        
        self.pre_process_data(reviews, labels)
        
        self.init_network(self.review_vocab_size, hidden_nodes, 1, learning_rate)
    
    # Process all the review and label data and form unique dictionarys
    # Give each word a unique index for entry into the network
    def pre_process_data(self, reviews, labels):
        # Creates a dictionary that contains one of each different word
        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):
                review_vocab.add(word)
        
        self.review_vocab = list(review_vocab)
        
        # Creates a dictionary that contains one of each label type [IN our case 'POSITIVE' or 'NEGATIVE']
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
                
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of each dictionary
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Give each word a uniquely identifying index
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        # Give each label an identifying index [in our case only 0 or 1]
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
    
    # Initialize all the network parameters
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes for all three layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # Initialise weights
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1, input_nodes))
    
    # Activation function
    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))
    
    # Reset the layer for the next pass and add the new words from the review
    def update_input_layer(self, review):
        # Clear the previous state and set to 0
        self.layer_0 *= 0
        
        for word in review.split(' '):
            if (word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] = 1
    
    # Get the neumerical reprisentation of the output
    def get_target_for_label(self, label):
        if label == 'POSITIVE':
            return 1
        else:
            return 0
    
    # Train the network
    def train(self, training_reviews, training_labels):
        
        # Check that the inputs match the outputs
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        # Log the start time
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            ## Forward Pass ##
            
            # Input Layer
            self.update_input_layer(review)
            
            # Hidden Layer
            layer_1 = self.layer_0.dot(self.weights_0_1)
            
            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
            
            ## Backwards Pass ##
            
            # Output Error
            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid(layer_2, True)
            
            # Hidden Error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error
            
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate
            
            if (np.abs(layer_2_error) < 0.5):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    # Test the network
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            
            if pred == testing_labels[i]:
                correct += 1
                
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                            + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    # Forward propogate        
    def run(self, review):
        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
```


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
```


```python
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:0.0% Speed(reviews/sec):0.0 #Correct:0 #Trained:1 Training Accuracy:0.0%
    Progress:10.4% Speed(reviews/sec):181.3 #Correct:1823 #Trained:2501 Training Accuracy:72.8%
    Progress:20.8% Speed(reviews/sec):193.3 #Correct:3798 #Trained:5001 Training Accuracy:75.9%
    Progress:31.2% Speed(reviews/sec):198.1 #Correct:5877 #Trained:7501 Training Accuracy:78.3%
    Progress:41.6% Speed(reviews/sec):200.7 #Correct:8019 #Trained:10001 Training Accuracy:80.1%
    Progress:52.0% Speed(reviews/sec):200.9 #Correct:10142 #Trained:12501 Training Accuracy:81.1%
    Progress:62.5% Speed(reviews/sec):200.8 #Correct:12279 #Trained:15001 Training Accuracy:81.8%
    Progress:72.9% Speed(reviews/sec):201.9 #Correct:14394 #Trained:17501 Training Accuracy:82.2%
    Progress:83.3% Speed(reviews/sec):202.4 #Correct:16565 #Trained:20001 Training Accuracy:82.8%
    Progress:93.7% Speed(reviews/sec):203.0 #Correct:18750 #Trained:22501 Training Accuracy:83.3%
    Progress:99.9% Speed(reviews/sec):203.3 #Correct:20070 #Trained:24000 Training Accuracy:83.6%

Instantly we can see that the network is now training successfully; attaining an accuracy upwards of 83.5%!

However the training process is still slow and we need to find remove inefficiencies from the code to make it run faster.


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):1081.% #Correct:848 #Tested:1000 Testing Accuracy:84.8%

## Analysing inefficiencies in the network

Now we need to identify things that can slow down the network. Lets start by looking at layer_0


```python
layer_0
```




    array([[ 18.,   0.,   0., ...,   0.,   0.,   0.]])



Notice that layer_0 is the length of the entire dictionary. However most of those inputs will simply be zero. Considering 0 * any number is always 0 it is a wasted calculation. Looking at a smaller example we can see...


```python
layer_0 = np.zeros(10)

layer_0
```




    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])




```python
layer_0[4] = 1
layer_0[9] = 1

layer_0
```




    array([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.])




```python
weights_0_1 = np.random.randn(10, 5)
layer_0.dot(weights_0_1)
```




    array([-0.10503756,  0.44222989,  0.24392938, -0.55961832,  0.21389503])



On its own this doesnt look like a problem but lets see how we can get the same results with much less computation.


```python
indices = [4,9]
layer_1 = np.zeros(5)
for index in indices:
    layer_1 += (weights_0_1[index])
    
layer_1
```




    array([-0.10503756,  0.44222989,  0.24392938, -0.55961832,  0.21389503])



See how this produces exactly the same results because 0 is a wasted value! It should also be noted that the input can only be a zero or a 1. 1 * x is always equal to x so the first layer computation can be significantly reduced!

## Removing inefficiencies in the network

Now that a serious inefficiency has been found in the network we need to actually adapt the network to handle it.


```python
import time
import sys

# Let's tweak the network from before to model these phenomena
class SentimentNetwork:
    def __init__(self, reviews,labels,hidden_nodes = 10, learning_rate = 0.1):
       
        np.random.seed(1)
    
        self.pre_process_data(reviews)
        
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)
        
        
    def pre_process_data(self,reviews):
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1,input_nodes))
        self.layer_1 = np.zeros((1,hidden_nodes))
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def update_input_layer(self,review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(" "):
            self.layer_0[0][self.word2index[word]] = 1

    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def train(self, training_reviews_raw, training_labels):
        
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))
        
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer

            # Hidden layer
#             layer_1 = self.layer_0.dot(self.weights_0_1)
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
            
            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            if(np.abs(layer_2_error) < 0.5):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
        
    
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                            + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        
        # Input Layer


        # Hidden layer
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
        
```


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
```


```python
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:99.9% Speed(reviews/sec):1998. #Correct:20099 #Trained:24000 Training Accuracy:83.7%


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):2020.% #Correct:852 #Tested:1000 Testing Accuracy:85.2%

## Further Noise Reduction

Lets continue to reduce the amount of noise to make training even more accurate! Small changes can make a huge impact on how fast the network trains.

Here we are going to look at how to seperate out the key words more effectively. 


```python
pos_neg_ratios.most_common()[0:30]
```




    [('edie', 4.6913478822291435),
     ('paulie', 4.0775374439057197),
     ('felix', 3.1527360223636558),
     ('polanski', 2.8233610476132043),
     ('matthau', 2.8067217286092401),
     ('victoria', 2.6810215287142909),
     ('mildred', 2.6026896854443837),
     ('gandhi', 2.5389738710582761),
     ('flawless', 2.451005098112319),
     ('superbly', 2.2600254785752498),
     ('perfection', 2.1594842493533721),
     ('astaire', 2.1400661634962708),
     ('captures', 2.0386195471595809),
     ('voight', 2.0301704926730531),
     ('wonderfully', 2.0218960560332353),
     ('powell', 1.9783454248084671),
     ('brosnan', 1.9547990964725592),
     ('lily', 1.9203768470501485),
     ('bakshi', 1.9029851043382795),
     ('lincoln', 1.9014583864844796),
     ('refreshing', 1.8551812956655511),
     ('breathtaking', 1.8481124057791867),
     ('bourne', 1.8478489358790986),
     ('lemmon', 1.8458266904983307),
     ('delightful', 1.8002701588959635),
     ('flynn', 1.7996646487351682),
     ('andrews', 1.7764919970972666),
     ('homer', 1.7692866133759964),
     ('beautifully', 1.7626953362841438),
     ('soccer', 1.7578579175523736)]



The data being passed at the moment is full of words we can use for sentiment analysis. However it also contains many words that are not usefull to us. There are many names that occur commonly that don't really express sentiment. The next step is to remove those useless words.

A good way to do this is to use bokeh; It is a data visualisation library, which will allow us to see what is going on with the data.


```python
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
output_notebook()
```



    <div class="bk-root">
        <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="fb46ab6e-4b43-42b8-8be9-d01fcb3c34cc">Loading BokehJS ...</span>
    </div>





```python
hist, edges = np.histogram(list(map(lambda x:x[1], pos_neg_ratios.most_common())), density=True, bins=100, normed=True)

p = figure(tools='pan,wheel_zoom,reset,save', toolbar_location='above', title="Word positive/negative Affinity Distribution")

p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
```




    <div class="bk-root">
        <div class="bk-plotdiv" id="cc88d3ef-17a0-4a9d-85fe-8eb4aa4f0334"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        document.getElementById("cc88d3ef-17a0-4a9d-85fe-8eb4aa4f0334").textContent = "BokehJS successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("cc88d3ef-17a0-4a9d-85fe-8eb4aa4f0334");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid 'cc88d3ef-17a0-4a9d-85fe-8eb4aa4f0334' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"7e445b23-4bc0-4dfa-b6c8-08386bc4d04c":{"roots":{"references":[{"attributes":{"formatter":{"id":"74e7aa9b-1532-4b13-8aa6-9f97d4963a86","type":"BasicTickFormatter"},"plot":{"id":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc","subtype":"Figure","type":"Plot"},"ticker":{"id":"b943d457-21c1-4b5e-a0dd-9c62ed3adf59","type":"BasicTicker"}},"id":"26861075-037c-412f-8379-1d74034aaa9e","type":"LinearAxis"},{"attributes":{"data_source":{"id":"5c90751c-1793-49bb-8456-c22501b19054","type":"ColumnDataSource"},"glyph":{"id":"880c3728-634d-4b36-b0af-32c19ef9bd2f","type":"Quad"},"hover_glyph":null,"nonselection_glyph":{"id":"a02f0734-eb22-4920-828c-b5717ecf8fae","type":"Quad"},"selection_glyph":null},"id":"f5b629c3-dbd2-4d6b-bae8-d3cfe3ac261f","type":"GlyphRenderer"},{"attributes":{},"id":"b943d457-21c1-4b5e-a0dd-9c62ed3adf59","type":"BasicTicker"},{"attributes":{"callback":null,"column_names":["left","right","top"],"data":{"left":{"__ndarray__":"Cvm3za5PEMCZHufvxesPwB5LXkQuOA/AonfVmJaEDsAnpEzt/tANwKzQw0FnHQ3AMf06ls9pDMC2KbLqN7YLwDpWKT+gAgvAv4KgkwhPCsBErxfocJsJwMnbjjzZ5wjATggGkUE0CMDSNH3lqYAHwFhh9DkSzQbA3I1rjnoZBsBhuuLi4mUFwObmWTdLsgTAahPRi7P+A8DwP0jgG0sDwHRsvzSElwLA+Zg2iezjAcB+xa3dVDABwAPyJDK9fADAED04DUuS/78Ylia2Gyv+vyLvFF/sw/y/LEgDCL1c+782ofGwjfX5v0D631lejvi/SFPOAi8n979SrLyr/7/1v1wFq1TQWPS/Zl6Z/aDx8r9wt4emcYrxv3gQdk9CI/C/BNPI8CV47b8YhaVCx6nqvyw3gpRo2+e/QOle5gkN5b9Qmzs4qz7iv8iaMBSZ4N6/8P7pt9tD2b8YY6NbHqfTv4COuf7BFMy/0FYsRkfbwL+AfHw2Moemv4BiuKu4XqY/QFB74yjRwD8AiAicowrMP+DfSioPotM/sHuRhsw+2T+QF9jiidveP7BZj58jPOI/oKeyTYIK5T+Q9dX74NjnP3hD+ak/p+o/aJEcWJ517T+o7x+D/iHwP6CWMdotifE/mD1DMV3w8j+M5FSIjFf0P4SLZt+7vvU/eDJ4Nusl9z9w2YmNGo34P2iAm+RJ9Pk/XCetO3lb+z9Uzr6SqML8P0h10OnXKf4/QBziQAeR/z+c4flLG3wAQBa1gveyLwFAkogLo0rjAUAMXJRO4pYCQIgvHfp5SgNABAOmpRH+A0B+1i5RqbEEQPqpt/xAZQVAdH1AqNgYBkDwUMlTcMwGQGwkUv8HgAdA5vfaqp8zCEBiy2NWN+cIQNye7AHPmglAWHJ1rWZOCkDSRf5Y/gELQE4ZhwSWtQtAyuwPsC1pDEBEwJhbxRwNQMCTIQdd0A1AOmeqsvSDDkC2OjNejDcPQDAOvAkk6w9A1nCi2l1PEECU2mawKakQQFJEK4b1AhFADq7vW8FcEUDMF7QxjbYRQIqBeAdZEBJASOs83SRqEkA=","dtype":"float64","shape":[100]},"right":{"__ndarray__":"mR7n78XrD8AeS15ELjgPwKJ31ZiWhA7AJ6RM7f7QDcCs0MNBZx0NwDH9OpbPaQzAtimy6je2C8A6Vik/oAILwL+CoJMITwrARK8X6HCbCcDJ24482ecIwE4IBpFBNAjA0jR95amAB8BYYfQ5Es0GwNyNa456GQbAYbri4uJlBcDm5lk3S7IEwGoT0Yuz/gPA8D9I4BtLA8B0bL80hJcCwPmYNons4wHAfsWt3VQwAcAD8iQyvXwAwBA9OA1Lkv+/GJYmthsr/r8i7xRf7MP8vyxIAwi9XPu/NqHxsI31+b9A+t9ZXo74v0hTzgIvJ/e/Uqy8q/+/9b9cBatU0Fj0v2Zemf2g8fK/cLeHpnGK8b94EHZPQiPwvwTTyPAleO2/GIWlQsep6r8sN4KUaNvnv0DpXuYJDeW/UJs7OKs+4r/ImjAUmeDev/D+6bfbQ9m/GGOjWx6n07+Ajrn+wRTMv9BWLEZH28C/gHx8NjKHpr+AYriruF6mP0BQe+Mo0cA/AIgInKMKzD/g30oqD6LTP7B7kYbMPtk/kBfY4onb3j+wWY+fIzziP6Cnsk2CCuU/kPXV++DY5z94Q/mpP6fqP2iRHFiede0/qO8fg/4h8D+gljHaLYnxP5g9QzFd8PI/jORUiIxX9D+Ei2bfu771P3gyeDbrJfc/cNmJjRqN+D9ogJvkSfT5P1wnrTt5W/s/VM6+kqjC/D9IddDp1yn+P0Ac4kAHkf8/nOH5Sxt8AEAWtYL3si8BQJKIC6NK4wFADFyUTuKWAkCILx36eUoDQAQDpqUR/gNAftYuUamxBED6qbf8QGUFQHR9QKjYGAZA8FDJU3DMBkBsJFL/B4AHQOb32qqfMwhAYstjVjfnCEDcnuwBz5oJQFhyda1mTgpA0kX+WP4BC0BOGYcElrULQMrsD7AtaQxARMCYW8UcDUDAkyEHXdANQDpnqrL0gw5AtjozXow3D0AwDrwJJOsPQNZwotpdTxBAlNpmsCmpEEBSRCuG9QIRQA6u71vBXBFAzBe0MY22EUCKgXgHWRASQEjrPN0kahJABlUBs/DDEkA=","dtype":"float64","shape":[100]},"top":{"__ndarray__":"s6auGMn1ZT+zpq4YyfVlPwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALOmrhjJ9WU/AAAAAAAAAAAAAAAAAAAAALOmrhjJ9WU/k6auGMn1ZT8AAAAAAAAAAJOmrhjJ9XU/AAAAAAAAAAAAAAAAAAAAAJOmrhjJ9WU/0qauGMn1dT+Tpq4YyfV1P7OmrhjJ9YU/Bf2C0lZ4gD+zpq4YyfWFP7OmrhjJ9WU/wdGY9Q83kz9fUNpeO3ObPwX9gtJWeJA/s6auGMn1lT/d0Zj1DzeTP8HRmPUPN6M/9GVPzd4Tqj+Je8Q7grSoP/RlT83eE6o/O3JIGwUosT9zD3sTUZGvP/RlT83eE7o/9GVPzd4Tuj+ht+X2LdDAP+gbdGF3pcY/2lQY73k5zz+nXNOsYYfSP6wBwWKVPtQ/xJV3OmQb2z9Z/6EadlviPyTgAZkQXOU/7sBhF6tc6D9YgiEU4F3uP2El8IH0Me4/XSz8TJet6j/wnIMFB5fnP/EzCBg6Kec/9UZePr7m4z89itzRx6vhP+ED4Kq0IdY/9UZePr7m0z+XmrXKouHOP/nrS/TxncU/Zbwjh2yWxD+hOmXwl9K8P/VGXj6+5rM/mzHpzxpGtT/8kDmqJVWnP2W8I4dslqQ/wdGY9Q83oz8qvCOHbJakP/jRmPUPN5M/wdGY9Q83kz8d/YLSVniQP5OmrhjJ9YU/k6auGMn1hT/Spq4YyfVlP5OmrhjJ9WU/0qauGMn1ZT8AAAAAAAAAAJOmrhjJ9WU/0qauGMn1ZT+Tpq4YyfVlP9KmrhjJ9WU/k6auGMn1dT8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADSpq4YyfVlPwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAk6auGMn1ZT8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAk6auGMn1ZT8=","dtype":"float64","shape":[100]}}},"id":"5c90751c-1793-49bb-8456-c22501b19054","type":"ColumnDataSource"},{"attributes":{"dimension":1,"plot":{"id":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc","subtype":"Figure","type":"Plot"},"ticker":{"id":"b943d457-21c1-4b5e-a0dd-9c62ed3adf59","type":"BasicTicker"}},"id":"acced3c4-41de-4dd5-af98-8ba56ffcd365","type":"Grid"},{"attributes":{"plot":{"id":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc","subtype":"Figure","type":"Plot"}},"id":"481b45bc-fb26-4a0b-be1c-97173aebc981","type":"PanTool"},{"attributes":{},"id":"a91d06a2-1a36-440f-a65d-391e38486885","type":"BasicTickFormatter"},{"attributes":{"plot":{"id":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc","subtype":"Figure","type":"Plot"}},"id":"5123d3a5-ffa9-4002-882d-71321d0a306e","type":"WheelZoomTool"},{"attributes":{"plot":{"id":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc","subtype":"Figure","type":"Plot"}},"id":"2721ebe5-714a-44e5-8cb0-8e8a58b4267b","type":"ResetTool"},{"attributes":{"plot":{"id":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc","subtype":"Figure","type":"Plot"}},"id":"4cce40c5-929e-453e-9fb4-fa5b93726baf","type":"SaveTool"},{"attributes":{"below":[{"id":"a078390d-84a5-4fb9-bdee-c1b68b2cbf92","type":"LinearAxis"}],"left":[{"id":"26861075-037c-412f-8379-1d74034aaa9e","type":"LinearAxis"}],"renderers":[{"id":"a078390d-84a5-4fb9-bdee-c1b68b2cbf92","type":"LinearAxis"},{"id":"a6e4ccd6-e0cc-4f10-a1ee-bae9d7f4d375","type":"Grid"},{"id":"26861075-037c-412f-8379-1d74034aaa9e","type":"LinearAxis"},{"id":"acced3c4-41de-4dd5-af98-8ba56ffcd365","type":"Grid"},{"id":"f5b629c3-dbd2-4d6b-bae8-d3cfe3ac261f","type":"GlyphRenderer"}],"title":{"id":"fb25d8cf-836d-4ca4-9886-e693de3e88db","type":"Title"},"tool_events":{"id":"51e33b7c-c6db-4b8b-b32f-39a417788c79","type":"ToolEvents"},"toolbar":{"id":"f2cd8bed-ec2e-4949-972a-2d9ba15218fa","type":"Toolbar"},"toolbar_location":"above","x_range":{"id":"4fea6e80-d155-4033-bc70-c061803cc245","type":"DataRange1d"},"y_range":{"id":"319b6fad-2e3d-42f3-a73d-75e03402ce01","type":"DataRange1d"}},"id":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc","subtype":"Figure","type":"Plot"},{"attributes":{"plot":null,"text":"Word positive/negative Affinity Distribution"},"id":"fb25d8cf-836d-4ca4-9886-e693de3e88db","type":"Title"},{"attributes":{},"id":"51e33b7c-c6db-4b8b-b32f-39a417788c79","type":"ToolEvents"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"481b45bc-fb26-4a0b-be1c-97173aebc981","type":"PanTool"},{"id":"5123d3a5-ffa9-4002-882d-71321d0a306e","type":"WheelZoomTool"},{"id":"2721ebe5-714a-44e5-8cb0-8e8a58b4267b","type":"ResetTool"},{"id":"4cce40c5-929e-453e-9fb4-fa5b93726baf","type":"SaveTool"}]},"id":"f2cd8bed-ec2e-4949-972a-2d9ba15218fa","type":"Toolbar"},{"attributes":{},"id":"74e7aa9b-1532-4b13-8aa6-9f97d4963a86","type":"BasicTickFormatter"},{"attributes":{"callback":null},"id":"4fea6e80-d155-4033-bc70-c061803cc245","type":"DataRange1d"},{"attributes":{"formatter":{"id":"a91d06a2-1a36-440f-a65d-391e38486885","type":"BasicTickFormatter"},"plot":{"id":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc","subtype":"Figure","type":"Plot"},"ticker":{"id":"fb53bc5f-385d-4a24-ae47-08d745787b96","type":"BasicTicker"}},"id":"a078390d-84a5-4fb9-bdee-c1b68b2cbf92","type":"LinearAxis"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"right":{"field":"right"},"top":{"field":"top"}},"id":"a02f0734-eb22-4920-828c-b5717ecf8fae","type":"Quad"},{"attributes":{"callback":null},"id":"319b6fad-2e3d-42f3-a73d-75e03402ce01","type":"DataRange1d"},{"attributes":{},"id":"fb53bc5f-385d-4a24-ae47-08d745787b96","type":"BasicTicker"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#1f77b4"},"left":{"field":"left"},"line_color":{"value":"#555555"},"right":{"field":"right"},"top":{"field":"top"}},"id":"880c3728-634d-4b36-b0af-32c19ef9bd2f","type":"Quad"},{"attributes":{"plot":{"id":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc","subtype":"Figure","type":"Plot"},"ticker":{"id":"fb53bc5f-385d-4a24-ae47-08d745787b96","type":"BasicTicker"}},"id":"a6e4ccd6-e0cc-4f10-a1ee-bae9d7f4d375","type":"Grid"}],"root_ids":["3d3a0acb-b2f1-48fd-900f-0c45eae16dbc"]},"title":"Bokeh Application","version":"0.12.4"}};
            var render_items = [{"docid":"7e445b23-4bc0-4dfa-b6c8-08386bc4d04c","elementid":"cc88d3ef-17a0-4a9d-85fe-8eb4aa4f0334","modelid":"3d3a0acb-b2f1-48fd-900f-0c45eae16dbc"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("cc88d3ef-17a0-4a9d-85fe-8eb4aa4f0334")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>


Here we can see that there are a large number of words lying in the middle of the x axis. This shows that there are a large number of words that have little meaning and less words on the sides with meaning.

This is good because we can add a cut-off point which will discount the useless words.

### Using the data to remove the noise

Now we can use what we have learned to increase the accuracy of the network by introducing a fequency cut-off. This will remove the useless words like '.' and ''.


```python
import time
import sys
import numpy as np

# Let's tweak the network from before to model these phenomena
class SentimentNetwork:
    def __init__(self, reviews,labels,min_count = 10,polarity_cutoff = 0.1,hidden_nodes = 10, learning_rate = 0.1):
       
        np.random.seed(1)
    
        self.pre_process_data(reviews, polarity_cutoff, min_count)
        
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)
        
        
    def pre_process_data(self,reviews, polarity_cutoff,min_count):
        
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if(labels[i] == 'POSITIVE'):
                for word in reviews[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term,cnt in list(total_counts.most_common()):
            if(cnt >= 50):
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word,ratio in pos_neg_ratios.most_common():
            if(ratio > 1):
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                if(total_counts[word] > min_count):
                    if(word in pos_neg_ratios.keys()):
                        if((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1,input_nodes))
        self.layer_1 = np.zeros((1,hidden_nodes))
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def update_input_layer(self,review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(" "):
            self.layer_0[0][self.word2index[word]] = 1

    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def train(self, training_reviews_raw, training_labels):
        
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))
        
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer

            # Hidden layer
#             layer_1 = self.layer_0.dot(self.weights_0_1)
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
            
            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            if(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
        
    
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                            + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        
        # Input Layer


        # Hidden layer
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] >= 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
```

min_count_cutoff: In order to be included it must be more frequent than this value
polarity_cutoff: Words must be left or right of the histogram by this much to be included


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.05,learning_rate=0.01)
```


```python
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:99.9% Speed(reviews/sec):2160. #Correct:20461 #Trained:24000 Training Accuracy:85.2%


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):2551.% #Correct:859 #Tested:1000 Testing Accuracy:85.9%

Reducing the useless data has clearly had a positive impact on the overall accuracy of the network during testing. It has also had a small increase in the speed because the dictionary is smaller. 

Increasing the polarity cutoff will speed up the network.


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.8,learning_rate=0.01)
```


```python
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:99.9% Speed(reviews/sec):7229. #Correct:20552 #Trained:24000 Training Accuracy:85.6%


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Progress:99.9% Speed(reviews/sec):4765.% #Correct:822 #Tested:1000 Testing Accuracy:82.2%

Here we got an increase in speed from 2236 w/s to 6251 w/s however 3% of the accuracy was traded off as a result. There will almost always be a trade-off between speed and accuracy excluding noise reduction which normally helps both.

## Analysis - Whats going on with the weights?

Here we will visualise what is realling going on under the hood of the network and how the weights work.


```python
mlp_full = SentimentNetwork(reviews[:-1000], labels[:-1000], min_count=0, polarity_cutoff=0, learning_rate=0.01)
```


```python
mlp_full.train(reviews[:-1000], labels[:-1000])
```

    Progress:99.9% Speed(reviews/sec):1804. #Correct:20335 #Trained:24000 Training Accuracy:84.7%


```python
def get_most_similar_words(focus = "horrible"):
    most_similar = Counter()

    for word in mlp_full.word2index.keys():
        most_similar[word] = np.dot(mlp_full.weights_0_1[mlp_full.word2index[word]],mlp_full.weights_0_1[mlp_full.word2index[focus]])
    
    return most_similar.most_common()
```


```python
get_most_similar_words('excellent')[0:50]
```




    [('excellent', 0.13672950757352473),
     ('perfect', 0.12548286087225946),
     ('amazing', 0.091827633925999713),
     ('today', 0.090223662694414231),
     ('wonderful', 0.089355976962214617),
     ('fun', 0.087504466674206888),
     ('great', 0.087141758882292031),
     ('best', 0.085810885617880639),
     ('liked', 0.07769762912384344),
     ('definitely', 0.076628781406966023),
     ('brilliant', 0.073423858769279038),
     ('loved', 0.073285428928122148),
     ('favorite', 0.072781136036160751),
     ('superb', 0.071736207178505068),
     ('fantastic', 0.070922191916266197),
     ('job', 0.069160617207634056),
     ('incredible', 0.06642407795261443),
     ('enjoyable', 0.065632560502888793),
     ('rare', 0.064819212662615075),
     ('highly', 0.063889453350970515),
     ('enjoyed', 0.062127546101812953),
     ('wonderfully', 0.062055178604090155),
     ('perfectly', 0.061093208811887401),
     ('fascinating', 0.060663547937493886),
     ('bit', 0.059655427045653034),
     ('gem', 0.059510859296156786),
     ('outstanding', 0.058860808147083013),
     ('beautiful', 0.058613934703162042),
     ('surprised', 0.058273314482562975),
     ('worth', 0.057657484236471199),
     ('especially', 0.057422020781760785),
     ('refreshing', 0.057310532092265769),
     ('entertaining', 0.056612033835629218),
     ('hilarious', 0.056168541032286662),
     ('masterpiece', 0.054993988649431544),
     ('simple', 0.054484083134924047),
     ('subtle', 0.054368883033508647),
     ('funniest', 0.053457164871302691),
     ('solid', 0.052903564743620658),
     ('awesome', 0.052489194202770428),
     ('always', 0.052260328525345269),
     ('noir', 0.051530194726406908),
     ('guys', 0.051109413645642685),
     ('sweet', 0.050818930317525997),
     ('unique', 0.050670162263589183),
     ('very', 0.05013299494852845),
     ('heart', 0.04994805849824361),
     ('moving', 0.049424601164379134),
     ('atmosphere', 0.048842500895912848),
     ('strong', 0.048570880631759197)]



Because these words are supposed to give a similar output the network sees them as similar and therefore have similar weights. We can visualise these clusters buy plotting them on a graph using T-SNE.


```python
import matplotlib.colors as colors

words_to_visualize = list()
for word, ratio in pos_neg_ratios.most_common(500):
    if(word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)
    
for word, ratio in list(reversed(pos_neg_ratios.most_common()))[0:500]:
    if(word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)
```


```python
pos = 0
neg = 0

colors_list = list()
vectors_list = list()
for word in words_to_visualize:
    if word in pos_neg_ratios.keys():
        vectors_list.append(mlp_full.weights_0_1[mlp_full.word2index[word]])
        if(pos_neg_ratios[word] > 0):
            pos+=1
            colors_list.append("#00ff00")
        else:
            neg+=1
            colors_list.append("#000000")
```


```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(vectors_list)
```


```python
p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="vector T-SNE for most polarized words")

source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:,0],
                                    x2=words_top_ted_tsne[:,1],
                                    names=words_to_visualize))

p.scatter(x="x1", y="x2", size=8, source=source,color=colors_list)

word_labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
p.add_layout(word_labels)

show(p)

# green indicates positive words, black indicates negative words
```

    /Users/max/anaconda/lib/python3.6/site-packages/bokeh/util/deprecation.py:34: BokehDeprecationWarning: 
    Supplying a user-defined data source AND iterable values to glyph methods is deprecated.
    
    See https://github.com/bokeh/bokeh/issues/2056 for more information.
    
      warn(message)
    /Users/max/anaconda/lib/python3.6/site-packages/bokeh/util/deprecation.py:34: BokehDeprecationWarning: 
    Supplying a user-defined data source AND iterable values to glyph methods is deprecated.
    
    See https://github.com/bokeh/bokeh/issues/2056 for more information.
    
      warn(message)





    <div class="bk-root">
        <div class="bk-plotdiv" id="431dd0c9-b460-488f-b329-bf72552c6a32"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        document.getElementById("431dd0c9-b460-488f-b329-bf72552c6a32").textContent = "BokehJS successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("431dd0c9-b460-488f-b329-bf72552c6a32");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid '431dd0c9-b460-488f-b329-bf72552c6a32' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"a71d9186-081f-4735-b10e-a42440bed059":{"roots":{"references":[{"attributes":{"formatter":{"id":"1250c23e-c515-4298-a204-c3dee3664f16","type":"BasicTickFormatter"},"plot":{"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"},"ticker":{"id":"296ea7e0-d9b9-4b34-9c2d-536801e12ecd","type":"BasicTicker"}},"id":"a0fcd5ec-151a-4c14-b506-dd8d52455051","type":"LinearAxis"},{"attributes":{},"id":"1250c23e-c515-4298-a204-c3dee3664f16","type":"BasicTickFormatter"},{"attributes":{},"id":"296ea7e0-d9b9-4b34-9c2d-536801e12ecd","type":"BasicTicker"},{"attributes":{"below":[{"id":"1a6cab84-4756-4c66-b429-b3c59e15b790","type":"LinearAxis"}],"left":[{"id":"a0fcd5ec-151a-4c14-b506-dd8d52455051","type":"LinearAxis"}],"renderers":[{"id":"1a6cab84-4756-4c66-b429-b3c59e15b790","type":"LinearAxis"},{"id":"30a42cc3-61c2-42f3-bf39-6df2c79959e2","type":"Grid"},{"id":"a0fcd5ec-151a-4c14-b506-dd8d52455051","type":"LinearAxis"},{"id":"65c152bb-6a92-4055-b104-de140d811ed5","type":"Grid"},{"id":"05523265-9e10-478a-be5c-fced408d95c6","type":"GlyphRenderer"},{"id":"b6aec15f-6b34-4f8e-970b-e221ba5855da","type":"LabelSet"}],"title":{"id":"b1a62e8e-500e-4943-a845-beddeea68ba3","type":"Title"},"tool_events":{"id":"6b30ed16-04bf-43b5-87db-37cf997e92c9","type":"ToolEvents"},"toolbar":{"id":"dd7151fb-d8d7-484c-a1bf-cfca93c3a34f","type":"Toolbar"},"toolbar_location":"above","x_range":{"id":"10f4d84c-11f4-44de-a545-92679213fa89","type":"DataRange1d"},"y_range":{"id":"f12a1ff3-e572-4561-800e-1332f80edf01","type":"DataRange1d"}},"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"},{"attributes":{"callback":null,"column_names":["x1","x2","names","fill_color","line_color"],"data":{"fill_color":["#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000"],"line_color":["#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#00ff00","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000"],"names":["edie","paulie","felix","polanski","matthau","victoria","mildred","gandhi","flawless","superbly","perfection","astaire","captures","voight","wonderfully","powell","brosnan","lily","bakshi","lincoln","refreshing","breathtaking","bourne","lemmon","delightful","flynn","andrews","homer","beautifully","soccer","elvira","underrated","gripping","superb","delight","welles","sadness","sinatra","touching","timeless","macy","unforgettable","favorites","stewart","sullivan","extraordinary","hartley","brilliantly","friendship","wonderful","palma","magnificent","finest","jackie","ritter","tremendous","freedom","fantastic","terrific","noir","sidney","outstanding","pleasantly","mann","nancy","marie","marvelous","excellent","ruth","stanwyck","widmark","splendid","chan","exceptional","tender","gentle","poignant","gem","amazing","chilling","fisher","davies","captivating","darker","april","kelly","blake","overlooked","ralph","bette","hoffman","cole","shines","powerful","notch","remarkable","pitt","winters","vivid","gritty","giallo","portrait","innocence","psychiatrist","favorite","ensemble","stunning","burns","garbo","barbara","philip","panic","holly","carol","perfect","appreciated","favourite","journey","rural","bond","builds","brilliant","brooklyn","von","recommended","unfolds","daniel","perfectly","crafted","prince","troubled","consequences","haunting","cinderella","alexander","emotions","boxing","subtle","curtis","rare","loved","daughters","courage","dentist","highly","nominated","tony","draws","everyday","contrast","cried","fabulous","ned","fay","emma","sensitive","smooth","dramas","today","helps","inspiring","jimmy","awesome","unique","tragic","intense","stellar","rival","provides","depression","shy","carrie","blend","hank","diana","adorable","unexpected","achievement","bettie","happiness","glorious","davis","terrifying","beauty","ideal","fears","hong","seasons","fascinating","carries","satisfying","definite","touched","greatest","creates","aunt","walter","spectacular","portrayal","ann","enterprise","musicals","deeply","incredible","mature","triumph","margaret","navy","harry","lucas","sweet","joey","oscar","balance","warm","ages","guilt","glover","carrey","learns","unusual","sons","complex","essence","brazil","widow","solid","beautiful","holmes","awe","vhs","eerie","lonely","grim","sport","debut","destiny","thrillers","tears","rose","feelings","ginger","winning","stanley","cox","paris","heart","hooked","comfortable","mgm","masterpiece","themes","danny","anime","perry","joy","lovable","mysteries","hal","louis","charming","urban","allows","impact","italy","gradually","lifestyle","spy","treat","subsequent","kennedy","loving","surprising","quiet","winter","reveals","raw","funniest","pleased","norman","thief","season","secrets","colorful","highest","compelling","danes","castle","kudos","great","baseball","subtitles","bleak","winner","tragedy","todd","nicely","arthur","essential","gorgeous","fonda","eastwood","focuses","enjoyed","natural","intensity","witty","rob","worlds","health","magical","deeper","lucy","moving","lovely","purple","memorable","sings","craig","modesty","relate","episodes","strong","smith","tear","apartment","princess","disagree","kung","adventure","columbo","jake","adds","hart","strength","realizes","dave","childhood","forbidden","tight","surreal","manager","dancer","studios","con","miike","realistic","explicit","kurt","traditional","deals","holds","carl","touches","gene","albert","abc","cry","sides","develops","eyre","dances","oscars","legendary","hearted","importance","portraying","impressed","waters","empire","edge","jean","environment","sentimental","captured","styles","daring","frank","tense","backgrounds","matches","gothic","sharp","achieved","court","steals","rules","colors","reunion","covers","tale","rain","denzel","stays","blob","maria","conventional","fresh","midnight","landscape","animated","titanic","sunday","spring","cagney","enjoyable","immensely","sir","nevertheless","driven","performances","memories","nowadays","simple","golden","leslie","lovers","relationship","supporting","che","packed","trek","provoking","strikes","depiction","emotional","secretary","influenced","florida","germany","brings","lewis","elderly","owner","streets","henry","portrays","bears","china","anger","society","available","best","bugs","magic","delivers","verhoeven","jim","donald","endearing","relationships","greatly","charlie","brad","simon","effectively","march","atmosphere","influence","genius","emotionally","ken","identity","sophisticated","dan","andrew","india","roy","surprisingly","sky","romantic","match","meets","cowboy","wave","bitter","patient","stylish","britain","affected","beatty","love","paul","andy","performance","patrick","unlike","brooks","refuses","award","complaint","ride","dawson","luke","wells","france","sports","handsome","directs","rebel","boll","uwe","seagal","unwatchable","stinker","mst","incoherent","unfunny","waste","blah","horrid","pointless","atrocious","redeeming","prom","drivel","lousy","worst","laughable","awful","poorly","wasting","remotely","existent","boredom","miserably","sucks","uninspired","lame","insult","godzilla","uninteresting","gadget","appalling","unconvincing","unintentional","horrible","amateurish","pathetic","idiotic","stupidity","cardboard","wasted","crap","insulting","tedious","dreadful","dire","badly","suck","worse","terrible","embarrassing","mess","garbage","pile","stupid","ashamed","vampires","worthless","dull","inept","avoid","wooden","forgettable","fulci","crappy","bat","unbelievably","whatsoever","excuse","rubbish","ridiculous","junk","flop","boring","turkey","shark","topless","ridiculously","useless","seed","ripped","embarrassed","rambo","costs","hideous","horrendous","bother","dumb","disjointed","plastic","horribly","fest","ludicrous","unintentionally","obnoxious","mildly","bland","mummy","annoying","amateur","bad","dinosaurs","unless","fails","mediocre","awake","clichd","clich","meaningless","disappointment","zombies","asleep","miscast","irritating","utter","disappointing","screaming","supposed","kidding","poor","apes","unbelievable","fake","dude","dracula","joke","clumsy","random","cheap","idiots","devoid","trite","wannabe","unbearable","alright","pretentious","scooby","sucked","senseless","bo","bin","coherent","idiot","toilet","doo","werewolf","cabin","generous","offensive","monkey","painfully","renting","lazy","disgusting","blame","walked","seconds","generic","cheese","sloppy","huh","retarded","trash","shelf","ugly","oh","slightest","explanation","failed","cringe","blatant","clue","bored","cgi","sat","paid","warn","painful","nowhere","bore","absurd","flies","paint","porn","paper","predictable","pseudo","repetitive","outer","brain","sorry","vampire","motivation","unrealistic","wrestling","overrated","aliens","halfway","save","santa","security","contrived","lacks","whale","gore","bunch","hype","flat","noise","below","plain","spending","bothered","annoyed","sounded","honestly","minutes","wreck","lesbian","chick","dollar","f","secondly","wanna","rat","errors","shallow","synopsis","breasts","gray","yeah","nonsense","unnecessary","swear","grave","ruined","somebody","elvis","mindless","terribly","continuity","hoping","ha","nudity","endless","decent","torture","rented","disaster","downright","ok","fat","unpleasant","figured","rip","throwing","attempt","weak","slap","jesus","christian","barely","apparently","implausible","nothing","clichs","credibility","bible","explained","presumably","celluloid","couldn","money","snake","hollow","load","sake","total","priest","supposedly","consists","zombie","bomb","ape","bottom","christ","unfortunately","bullets","grade","drags","freak","wolf","fx","offended","script","raped","producers","okay","confusing","stomach","monster","seriously","alas","promising","knife","substance","premise","threw","k","dear","z","write","rental","warned","zero","semi","guess","scientist","logic","vague","slasher","throw","accents","alien","silly","clown","skip","instead","blank","throat","lab","par","gag","execution","nose","hated","effort","shoot","fill","gratuitous","burn","none","cameras","assume","stick","reasonable","failure","pie","rent","dubbing","weren","truck","stock","thin","daddy","holy","exercise","pg","arm","tried","suppose","advice","gonna","disbelief","derek","mean","merit","looked","channel","gross","stereotypical","hoped","lacking","spent","stiff","overdone","low","romero","hour","blair","saved","damage","reason","intentions","sentence","hardcore","makeup","lack","makers","empty","holes","wouldn","proof","demon","toys","doll","utterly","originality","bush","saying","cover","meat","forest","deserve","sum","bucks","hills","watchable","lacked","handed","mistake","please","whoever","sadistic","monsters","screenwriter","neither","nuclear","sequels","flesh","lying","creature","annie","propaganda","leonard","thats","racist","convince","asian","why","rex","satan","remake","fail","ah","loser","favor","except","flick","freddy","relies","spare","dialog","lou","dragged","guy","problem","melting","flash","im","least","mouth","sole","hell","jerk","drink","intent","shower","fifteen","wasn","thugs","corpse","virus","idea","budget","minimal","reasonably","naked","rick","category","cheesy","judging","half","pregnant","no","millions","stereotypes","juvenile","weekend","convoluted","laurel","killings","sequel","hire","somewhere","frankly","paying","someone","cant","cash","research","dimensional","walk","editing","conceived","scare","positive","anything"],"x1":{"__ndarray__":"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","dtype":"float64","shape":[1000]},"x2":{"__ndarray__":"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","dtype":"float64","shape":[1000]}}},"id":"3ddbb301-7cb2-4edd-aed7-b77feb56f67a","type":"ColumnDataSource"},{"attributes":{"callback":null},"id":"f12a1ff3-e572-4561-800e-1332f80edf01","type":"DataRange1d"},{"attributes":{},"id":"9fc50889-5464-4d39-995d-eb976161d778","type":"BasicTickFormatter"},{"attributes":{"plot":null,"text":"vector T-SNE for most polarized words"},"id":"b1a62e8e-500e-4943-a845-beddeea68ba3","type":"Title"},{"attributes":{},"id":"829f4a6d-049b-4010-94b7-d587a0c51dd0","type":"BasicTicker"},{"attributes":{"plot":{"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"}},"id":"fc5cb3e4-da44-4816-8f03-f186c34959da","type":"ResetTool"},{"attributes":{},"id":"6b30ed16-04bf-43b5-87db-37cf997e92c9","type":"ToolEvents"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"b39cc0dd-b053-4d51-bdc6-39cc7bc3f5fe","type":"PanTool"},{"id":"e1eae6f4-b285-40c3-a97b-e94215d6b263","type":"WheelZoomTool"},{"id":"fc5cb3e4-da44-4816-8f03-f186c34959da","type":"ResetTool"},{"id":"78ea573a-1253-437a-b777-16ad7c5fc3d3","type":"SaveTool"}]},"id":"dd7151fb-d8d7-484c-a1bf-cfca93c3a34f","type":"Toolbar"},{"attributes":{"plot":{"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"}},"id":"78ea573a-1253-437a-b777-16ad7c5fc3d3","type":"SaveTool"},{"attributes":{"formatter":{"id":"9fc50889-5464-4d39-995d-eb976161d778","type":"BasicTickFormatter"},"plot":{"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"},"ticker":{"id":"829f4a6d-049b-4010-94b7-d587a0c51dd0","type":"BasicTicker"}},"id":"1a6cab84-4756-4c66-b429-b3c59e15b790","type":"LinearAxis"},{"attributes":{"plot":{"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"},"ticker":{"id":"829f4a6d-049b-4010-94b7-d587a0c51dd0","type":"BasicTicker"}},"id":"30a42cc3-61c2-42f3-bf39-6df2c79959e2","type":"Grid"},{"attributes":{"dimension":1,"plot":{"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"},"ticker":{"id":"296ea7e0-d9b9-4b34-9c2d-536801e12ecd","type":"BasicTicker"}},"id":"65c152bb-6a92-4055-b104-de140d811ed5","type":"Grid"},{"attributes":{"fill_color":{"field":"fill_color"},"line_color":{"field":"line_color"},"size":{"units":"screen","value":8},"x":{"field":"x1"},"y":{"field":"x2"}},"id":"5caf80c8-27ae-4555-80ae-fcf86ae22fbf","type":"Circle"},{"attributes":{"plot":{"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"}},"id":"b39cc0dd-b053-4d51-bdc6-39cc7bc3f5fe","type":"PanTool"},{"attributes":{"callback":null},"id":"10f4d84c-11f4-44de-a545-92679213fa89","type":"DataRange1d"},{"attributes":{"plot":{"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"}},"id":"e1eae6f4-b285-40c3-a97b-e94215d6b263","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"3ddbb301-7cb2-4edd-aed7-b77feb56f67a","type":"ColumnDataSource"},"glyph":{"id":"5caf80c8-27ae-4555-80ae-fcf86ae22fbf","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"c88cd9e4-6e22-4140-9ad5-d34178b38b55","type":"Circle"},"selection_glyph":null},"id":"05523265-9e10-478a-be5c-fced408d95c6","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":8},"x":{"field":"x1"},"y":{"field":"x2"}},"id":"c88cd9e4-6e22-4140-9ad5-d34178b38b55","type":"Circle"},{"attributes":{"plot":{"id":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b","subtype":"Figure","type":"Plot"},"source":{"id":"3ddbb301-7cb2-4edd-aed7-b77feb56f67a","type":"ColumnDataSource"},"text":{"field":"names"},"text_align":"center","text_color":{"value":"#555555"},"text_font_size":{"value":"8pt"},"x":{"field":"x1"},"y":{"field":"x2"},"y_offset":{"value":6}},"id":"b6aec15f-6b34-4f8e-970b-e221ba5855da","type":"LabelSet"}],"root_ids":["fd87fb36-4a56-477c-b8b7-107ab7e5f56b"]},"title":"Bokeh Application","version":"0.12.4"}};
            var render_items = [{"docid":"a71d9186-081f-4735-b10e-a42440bed059","elementid":"431dd0c9-b460-488f-b329-bf72552c6a32","modelid":"fd87fb36-4a56-477c-b8b7-107ab7e5f56b"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("431dd0c9-b460-488f-b329-bf72552c6a32")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>


## Conclusion

After curating the dataset and training we were able to achieve a speed of a couple of hundred words per second with a very low accuracy. However after doing the first instance of noise reduction we were able to get the accuracy upwards of 80%. After optimising inefficiencys the network managed to achieve the same accuracy and a couple of thousand words per second. Finally a second set of noise reduction removed the useless data from the set such as names and punctuation. This acieved an even higher accuracy. After this step the network now has the ability to trade of accuracy for speed by cutting more words out of the vocabulary. This can be helpful for training over a much larger dataset. It also marginally increased speed by removing some of the data.

After visualising the data it can clearly be seen that the network has successfully grouped the input words by sentiment. With a few agnostic variables normally consisting of names that slipped through. This could be imporoved by increading the cutoff.
