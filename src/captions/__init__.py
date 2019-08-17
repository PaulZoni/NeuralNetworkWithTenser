from itertools import permutations

'''
possible_combinations = [
['1234', '4123', '3412', '2341'],
['1324', '4213', '3142', '2431'],
['1342', '4123', '3214', '2431'],
['1432', '4213', '3124', '2341'],
['1234', '3142', '2413', '4321'],
['1234', '3412', '2143', '4321'],
['2431', '3124', '1243', '4312'],
['4231', '1342', '3124', '2413'],
['4132', '2314', '3241', '1423'],
['4231', '2314', '3142', '1423'],
['4132', '2341', '3214', '1423'],
['2134', '3241', '1423', '4312'],
['4132', '2314', '3421', '1243'],
['1342', '4213', '3421', '2134'],
['1342', '4213', '3124', '2431'],
['4123', '1432', '2314', '3241'],
['1432', '4123', '3214', '2341'],
['4231', '1324', '3412', '2143'],
['4231', '1324', '3142', '2413'],
['1432', '3214', '2143', '4321'],
['4132', '1324', '3241', '2413'],
['1234', '3421', '2143', '4312'],
['2134', '3412', '1243', '4321'],
['2134', '3421', '1243', '4312']]
'''

SIZE = 4
EMPTY = '.'
towers = range(1, SIZE + 1)
'''
# generate sets of visible towers
visible = {}

for p in permutations(towers, SIZE):
    # from left
    max_visible, left = 0, 0
    for t in p:
        if t > max_visible:
            max_visible = t
            left += 1

    # from right
    max_visible, right = 0, 0
    for t in reversed(p):
        if t > max_visible:
            max_visible = t
            right += 1

    line = ''.join(map(str, p))
#    print(left, line, right)     #### DEBUG

    if (left, right) in visible:
        visible[(left, right)].add(line)
    else:
        visible[(left, right)] = {line}

#for p in visible: print(p, visible[p])     #### DEBUG
'''
visible = {
    (1, 2): {'4123', '4213'},
    (1, 3): {'4132', '4312', '4231'},
    (1, 4): {'4321'},
    (2, 1): {'3214', '3124'},
    (2, 2): {'1423', '3142', '3412', '2143', '2413', '3241'},
    (2, 3): {'1432', '3421', '2431'},
    (3, 1): {'2314', '1324', '2134'},
    (3, 2): {'2341', '1243', '1342'},
    (4, 1): {'1234'}}


def solve_puzzle(*clues):
    """generic solver for skycrapers puzzle"""

    def print_grid(title='grid:'):
        """print grid in a nice format"""
        print(title)
        for left, right, line in clue_rows:
            print(' '.join(line))

    def find_common():
        found = 0
        combos = visible[left, right]
        possible = [set(nums) for nums in zip(*combos)]
        for i in range(SIZE):
            if line[i] == EMPTY and len(possible[i]) == 1:
                line[i] = possible[i].pop()
                found += 1
        return found

    # restructure clues
    # top and bottom clues
    clue_cols = []
    for i in range(SIZE):
        top, bottom = i, SIZE * 3 - 1 - i
        clue_cols.append([clues[top], clues[bottom], [EMPTY] * SIZE])

    # left and right clues
    clue_rows = []
    for i in range(SIZE):
        left, right = SIZE * 4 - 1 - i, SIZE + i
        clue_rows.append([clues[left], clues[right], [EMPTY] * SIZE])

    #### DEBUG
    print('clues:')
    # for n, line in zip(clues, clue_line): print(n, line)
    for c in clue_rows + clue_cols:
        print(c)
    print_grid()  #### DEBUG

    # start solving!
    found = 0

    # while found < SIZE**2:
    for _ in range(2):
        # check rows
        for left, right, line in clue_rows:
            if EMPTY in line:
                found += find_common()

        # update columns
        temp_grid = [line for left, right, line in clue_rows]
        for i, line in enumerate(zip(*temp_grid)):
            clue_cols[i][-1] = list(line)

        # check columns
        for left, right, line in clue_cols:
            if EMPTY in line:
                found += find_common()

        # update rows
        temp_grid = [line for left, right, line in clue_cols]
        for i, line in enumerate(zip(*temp_grid)):
            clue_rows[i][-1] = list(line)

        print_grid()
        print('numbers found:', found)

    ###

    '''
    print('result:')
    for r in grid: print(r)     #### DEBUG
    tuple(tuple(r) for r in grid)
    '''
    return


solve_puzzle(2, 2, 1, 3,
             2, 2, 3, 1,
             1, 2, 2, 3,
             3, 2, 1, 3)
''' outcome:
( ( 1, 3, 4, 2 ),       
  ( 4, 2, 1, 3 ),       
  ( 3, 4, 2, 1 ),
  ( 2, 1, 3, 4 ) )
'''

'''
solve_puzzle( 0, 0, 1, 2,
              0, 2, 0, 0,
              0, 3, 0, 0,
              0, 1, 0, 0 )
'''
''' outcomes:
( ( 2, 1, 4, 3 ), 
  ( 3, 4, 1, 2 ), 
  ( 4, 2, 3, 1 ), 
  ( 1, 3, 2, 4 ) )
'''
