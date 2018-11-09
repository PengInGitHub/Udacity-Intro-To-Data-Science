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
