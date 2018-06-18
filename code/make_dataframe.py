"""LOads the Twitter json files for each user and creates CSV files."""
import argparse
import gensim
import glob
import gzip
import pandas
import os
import time
import ujson as json
import random
import re


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', help='where to look for input files')
parser.add_argument('--outdir', help='where to write output')
args = parser.parse_args()

# regex pattern to detect non-alphanumeric chars                  
pattern = re.compile('[^a-zA-Z\d\s#@]+')
punct_regex = re.compile(r'[\n\r\.\?!,:\"]+')

# phraser model to find bigrams
phraser = gensim.models.phrases.Phraser.load('../models/phraser.bin')


filenames = glob.glob(os.path.join(args.datadir, '*/*.gz'))
random.shuffle(filenames)
print('num files {0}'.format(len(filenames)))


def SaveRows(rows, filename):
  df = pandas.DataFrame(rows)
  df.to_csv(filename, index=None, encoding='utf8')


count = 0
rows = []
for idx, filename in enumerate(filenames):
    print(idx)
    try:
      with gzip.open(filename, 'r') as f:
        data = json.load(f)
    except e:
      print(e)
      print('bad file {0}'.format(filename))
      continue
    for tweet in data:
        timestamp = time.mktime(time.strptime(tweet['created_at'],"%a %b %d %H:%M:%S +0000 %Y"))
        text = punct_regex.sub(' ', tweet['text']).lower()
        text = ' '.join(phraser[text.split()])
        rows.append({'text': text, 'lang': tweet['lang'],
                     'user': tweet['user']['id'], 'timestamp': timestamp,
                     'dir': os.path.dirname(filename),
                     'file': os.path.basename(filename)})
                     
    if len(rows) > 10000000:  # ten million
        savename = os.path.join(args.outdir, 'tweet_df_{0}.csv'.format(count))
        SaveRows(rows, savename)
        count += 1
        rows = []

if len(rows) > 0:
  savename = os.path.join(args.outdir, 'tweet_df_{0}.csv'.format(count))
  SaveRows(rows, savename)
