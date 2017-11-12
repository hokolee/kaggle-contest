import sys


def read_ans_data(ans_file):
  with open(ans_file) as file:
    data = []
    for line in file: data.append(line)
    file.close()
    return data


print sys.argv
ans = read_ans_data(sys.argv[1])
output_file = sys.argv[2]
thr = float(sys.argv[3])

with open(output_file, 'wb') as csv_file:
  first = True
  for line in ans:
    if first:
      csv_file.write(line)
      first = False
    else:
      item = line.split(',')
      if float(item[1]) < thr:
        csv_file.write("%s,%.15lf\n" % (item[0], thr))
      else:
        csv_file.write("%s,%.15lf\n" % (item[0], float(item[1])))
  csv_file.close()
