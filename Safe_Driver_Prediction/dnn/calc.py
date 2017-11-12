import sys
import math

def read_ans_data(ans_file):
  with open(ans_file) as file:
    data = []
    for line in file: data.append(line)
    file.close()
    return data


print sys.argv
ans = sys.argv[1]
thr = float(sys.argv[2])

log_loss = 0.0
count = 0.0

with open(ans) as file:
  first = True
  for line in file:
    if first:
      first = False
    else:
      item = line.split(',')
      instance_id = item[0]
      prob = float(item[1])
      if prob < thr:
        prob = thr
      label = float(item[2])
      log_loss += label * math.log(prob) + (1 - label) * math.log(1 - prob)
      count += 1

log_loss = -1.0 * log_loss / count
print("log_loss: %.15f\n" % log_loss)
