import sys

def read_ans_data(ans_file):
  with open(ans_file) as ans:
    data = []
    for line in sorted(ans): data.append(line.strip().split(","))
    ans.close()
    return data

print >> sys.stderr, sys.argv

test_set = []
for i in range(1, len(sys.argv)):
  test = read_ans_data(sys.argv[i])
  test_set.append(test)

num = len(test_set)
length = len(test_set[0])

print("id,target")
for i in range(0, length):
  ans = 0.0
  for j in range(0, num):
    assert len(test_set[j][i]) == len(test_set[0][i])
    pro = float(test_set[j][i][1])
    ans += (1.0 / num) * pro
  result = "%s,%.4f" % (test_set[0][i][0], float(ans))
  print(result)

