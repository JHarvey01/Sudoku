from classes import number_square
DEFAULT_FILEPATH = 'tmp/sudoku_1.csv'

def _read_csv(filepath=DEFAULT_FILEPATH):
    """
    Reads a CSV file containing Sudoku puzzle data.

    Args:
        filepath (str): The path to the CSV file. Defaults to 'img/sudoku_1.csv'.

    Returns:
        list: A list of lists representing the Sudoku puzzle.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    header = lines[0].strip().split(',')
    puzzle = []
    for line in lines[1:]:
        row = []
        for value in line.strip().split(','):
            if value == '_':
                row.append(number_square())
            else:
                row.append(number_square(int(value)))
        puzzle.append(row)
    return puzzle, header

def _print_puzzle(puzzle):
    """
    Prints the Sudoku puzzle in a formatted way.

    Args:
        puzzle (list): A list of lists representing the Sudoku puzzle.
    """
    print('┌───────┬───────┬───────┐')
    for row in puzzle:
        print('│', end=' ')
        for square in row:
            if square.number == -1:
                print('-', end=' ')
            else:
                print(f'{square.number}', end=' ')
            if row.index(square) % 3 == 2:
                print('│', end=' ')
        print()
        if puzzle.index(row) % 3 == 2 and puzzle.index(row) != len(puzzle) - 1:
            print('├───────┼───────┼───────┤')
        elif puzzle.index(row) == len(puzzle) - 1:
            print('└───────┴───────┴───────┘')


if __name__ == "__main__":
    # Example usage
    sudoku_data, headers = _read_csv()
    
    print("Headers:", headers)
    _print_puzzle(sudoku_data)
    
    
