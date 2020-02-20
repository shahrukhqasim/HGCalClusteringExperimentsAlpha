import numpy as np
import matplotlib.pyplot as plt



a = []
b = []

a = np.array([  22,  111,  365,  714, 1396, 2245, 3156, 4128, 5242, 6073, 6937,
       7591, 8168, 8626, 8812, 8923, 8749, 8625, 8171, 7912, 7188, 6719,
       6247, 5673, 5015, 4623, 4076, 3598, 3098, 2714, 2335, 2072, 1752,
       1444, 1258,  956,  852,  744,  603,  475,  392,  336,  291,  198,
        153,  156,  110,   92,   87,   58,   42,   42,   34,   17,   27,
         20,    7,    8,    8,    5,    5,    5,    0,    2,    0,    1,
          2,    0,    1,    1,    1,    0,    0,    0,    0])

b = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
               68, 69, 70, 71, 72, 73, 74])


a = 100*a/np.sum(a)

print(a.shape, b.shape)



plt.bar(b, a)
plt.title('')
plt.xlabel('Number of showers')
plt.ylabel('Frequency (%)')
plt.title('Distribution of number of showers')
plt.show()