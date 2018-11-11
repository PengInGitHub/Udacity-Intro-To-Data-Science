x = [1,2,3]
x.extend([3,4,5])
print(x)

# list addition
y = x + [2,3,4,5]
print(x, y)

# append item to list
x.append(8)
print(x)

# unpack list
a, b = [1,2]
print(a, b)

# count frequency - use dict
word_counts = {}
doc = ['I','think','I','like','like','I']

# normal
for word in doc:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# try ... exception
for word in doc:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1

# use get
for word in doc:
    pre_val = word_counts.get(word, 0)
    word_counts[word] = pre_val + 1

# use defaultdict
from collections import defaultdict
word_counts = defaultdict(int) # default is zero
for word in doc:
    word_counts[word] += 1

# defaultdict practice
dd_list = defaultdict(list)
dd_list[2].append(1)

dd_dict = defaultdict(dict)
dd_dict['Joe']['City'] = 'New York'

# Counter
from collections import Counter
c = Counter([0,0,0,1,2])
# c is a defaultdict - like object
print(c)
c = Counter(doc)
print(c.most_common(2))

# ternary if-then-else
for i in range(11):
    #print(i)
    p = 'even' if i%2 == 0 else 'odd'
    print(p)

# continue and break
for i in range(10):
    #continue if i == 3
    if i == 3:
        continue
    elif i == 5:
        break
    print(i)

# if obj is not empty
s = ""
if s:
    print(' none')
else:
    print('is none')

# and: and returns its 2nd val if the 1st is truthy
s = None
first_char = s and s[0]
# object of type 'NoneType' has no len()
print(len(first_char))
if s:
    print('not none')
else:
    print(' none')


all([]) # True
any([])

wc = sorted(word_counts.items(), key=lambda(word,count):count, reverse=True)

# list comprehension
evens = [x for x in range(5) if x % 2 == 0]
squares = [x**2 for x in range(10)]
even_sqr = [x**2 for x in evens]

# turn list into dict or set
squar_dict = {x:x**2 for x in range(5)}
print(squar_dict)
square_set = {x*x for x in range(-1,2)}
print(square_set)

# use _ to avoid not-in-use var
zeros = [0 for _ in evens]
print(zeros)

# a list comprehension can include multiple fors
res = []
for i in range(10):
    for j in range(10):
        res.append((i,j))
print(res)

# list comprehensipn
pairs = [(x,y)
        for x in range(10)
        for y in range(10)]
print(pairs)

# randomness
import random
four_uniform_randoms = [random.random() for _ in range(4)]
print(four_uniform_randoms)

# use seed to get reproducible res:
random.seed(10)
print(random.random())

print(random.randrange(10))
print(random.randrange(5,10))

up_to_ten = range(10)
random.shuffle(up_to_ten)
print(up_to_ten)

# choose one ele from a list
best = random.choice(['FB', 'Ali', 'Ten']) 
print(best)

# choose a sample of nums from a list without replacement
# aka no duplicates
# use random.sample()
lottery = random.sample(range(60), 6)
print(lottery)

# with duplicates
# use random.choice() for multiple times
with_replacement = [random.choice(range(60)) for _ in range(10)]
print(with_replacement)


### Regular Expression

# Functional Tools
# map, reduce, filter
def double(x):
    return 2*x
xs = [4,5,6,7,8]
# list comprehension
twice_xs = [double(x) for x in xs]
# map
twice_xs = map(double, xs)
print(twice_xs)

# filter does for work of list comprehension if
def is_even(x):
    return x % 2 == 0
x_is_even = [x for x in xs if is_even(x)]
x_is_even = filter(is_even, xs)
print(x_is_even)

# reduce
help(reduce)
# reduce the sequence to a single value

def multiply(x, y): return x*y
print(multiply(3,4))

x_product = reduce(multiply, xs)
print(x_product)

for i,v in enumerate(xs):
    print( i, v)


# zip(): zip lists together
# return list of tuples of corresponding elements
list1 = [1,2,3]
list2 = ['a','b','c']
a = zip(list1, list2)
print(a)

# unzip
num, letter = zip(*a)
print(num)
print(letter)

# use argument unpacking anawhere
print(multiply(*[3,4]))

# visualization
from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# create a line chart, years on x, gdp on y
plt.plot(years, gdp,marker='o',linestyle='solid')
# add title
plt.title('Nominal GDP')
plt.ylabel('Billions of $')
plt.show()

# bar charts
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

xs = [x+0.1 for x,_ in enumerate(movies)]
print(xs)

# plt bars
plt.bar(xs, num_oscars)
plt.ylabel('# of Academy Awards')
plt.title('selected movies')
plt.xticks([i+0.1 for i,_ in enumerate(movies)],movies)
plt.show()

# bar chart is good for plotting histgrams of buckted num value
# to explore how these values are distributed
grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
decile = lambda grade: grade // 10 * 10
print(decile)
histogram = Counter(decile(grade) for grade in grades)
pos = [x for x in histogram.keys()]
print(pos)

plt.bar(pos, histogram.values(), 8)
plt.axis([-5,105,0,5])
plt.xticks([10 * i for i in range(11)])
plt.xlabel('Decile')
plt.ylabel('# of students')
plt.title('Distribution of Exam 1 Grades')
plt.show()


# Line charts
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x+y for x, y in zip(variance,bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# make duplicate calls to plt.plot()
print(xs)
plt.plot(xs, variance, 'g-', label='variance')
plt.plot(xs, bias_squared, 'r-', label='bias^2')
plt.plot(xs, total_error, 'b:', label='total error')
plt.legend(loc=9)
plt.xlabel('model complexity')
plt.title('The Bias Variance Tradeoff')
plt.show()

# Scatter Plot

friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# label each point
for label, friend_count, min_count in zip(labels, friends, minutes):
    plt.annotate(label,
                 xy=(friend_count,min_count),
                 xytext=(5,-5),
                 textcoords = 'offset points')

plt.title('Daily Minutes vs. Number of Friends')
plt.xlabel('# of friends')
plt.ylabel('daily minutes spent on the site')
plt.show()


test_1_grades = [ 99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.axis('equal')
plt.title('Axes are not comparable')
plt.xlabel('test 1 grade')
plt.ylabel('test 2 grade')
plt.show()

# histogram
import math

def bucketize(point, bucket_size):
    return bucket_size * math.floor(point/bucket_size)

def make_histogram(points, bucket_size):
    # bucketize the points and count how many in each bucket
    return Counter(bucketize(point, bucket_size) for point in points)


def plot_histogram(points, bucket_size, title=''):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(),width=bucket_size)
    plt.title(title)
    plt.show()


def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""
    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z, low_p = -10.0, 0 # normal_cdf(-10) is (very close to) 0
    hi_z, hi_p = 10.0, 1 # normal_cdf(10) is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2 # consider the midpoint
        mid_p = normal_cdf(mid_z) # and the cdf's value there
        if mid_p < p:
        # midpoint is still too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
        # midpoint is still too high, search below it
            hi_z, hi_p = mid_z, mid_p
        else:
            break
    return mid_z


random.seed(0)
# uniform between -100 t0 100
uniform = [200 * random.random() - 100 for _ in range(100000)]

normal = [57 * inverse_normal_cdf(random.random())
         for _ in range(100000)]


# uniform and normal have very diff distributions
plot_histogram(uniform, 10, 'Uniform Histogram')



























