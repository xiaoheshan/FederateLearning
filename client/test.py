def load_text(path):
    with open(path,'r') as f:
        return f.read()

def get_sum_time(strArr):
    sum = 0
    for str in strArr:
        sum += float(str)
    return sum

times = []
for i in range(10):
    timeArr = load_text("./time/Thread{}.txt".format(i))
    timeArr = timeArr.split(" ")
    timeArr.remove("")
    print(timeArr)
    times.append(get_sum_time(timeArr))

print(times)