import argparse

parser = argparse.ArgumentParser(description='parse filename')
parser.add_argument('--ans', type=str, help='answer')
parser.add_argument('--res', type=str, help='result')

args = parser.parse_args()

MY_RESULT = args.res
GIVEN_RESULT = args.ans

l1 = []
r2 = set()

with open(MY_RESULT, 'r') as f:
    for line in f:
        for token in line.strip().split(' '):
            l1.append(int(token))

with open(GIVEN_RESULT, 'r') as f:
    for line in f:
        for token in line.strip().split(' '):
            r2.add(int(token))

get_them = set(l1) & r2  # intersection

def f_score(precision, recall):
    return 1 / (1 / precision + 1 / recall)

precisions = []
recalls = []
fscores = []
count = 0
for i in range(len(l1)):
    if count == len(get_them):
        break
    if l1[i] in r2:
        count += 1
        
        precision = count / (i + 1)
        recall = count / len(r2)

        precisions.append(precision)
        recalls.append(recall)
        fscores.append(f_score(precision, recall))

print('precisions:   ', precisions)
print('recalls:      ', recalls)
print('F-scores:     ', fscores)
print('avg precision:   ', sum(precisions) / len(fscores))
print('avg recall:      ', sum(recalls) / len(fscores))
print('avg F-score:     ', sum(fscores) / len(fscores))

