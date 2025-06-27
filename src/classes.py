class number_square:
    def __init__(self, number=-1):
        self.number = number
        self.options = []

class sudoku_puzzle:
    def __init__(self):
        self.puzzle = [[number_square() for _ in range(9)] for _ in range(9)]
    
    def print_puzzle(self):
        """
        Prints the Sudoku puzzle in a nicely formatted grid using the numbers from number_square objects.
        """
        print('┌───────┬───────┬───────┐')
        for i, row in enumerate(self.puzzle):
            print('│', end=' ')
            for j, square in enumerate(row):
                val = '-' if square.number == -1 else str(square.number)
                print(val, end=' ')
                if (j + 1) % 3 == 0:
                    print('│', end=' ')
            print()
            if (i + 1) % 3 == 0 and i != 8:
                print('├───────┼───────┼───────┤')
            elif i == 8:
                print('└───────┴───────┴───────┘')