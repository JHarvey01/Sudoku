# Sudokuproject

> For reading in images of sudoku for step by step solving and help.

## 🧠 Project Overview

The goal of this project is to use Machine Learning and computer vision to read in sudoku puzzles which then algorithmically go through a set of solving rules from simple to hard and  solve puzzles step-by-step. The goal is to eventually be able to take a picture of the puzzle and either get the next number to get unstuck or get a hint at the rule that shold be applied for the next step.

Note that initial versions should not handle handwritten digits.

## 📦 Project Structure

| Folder/File       | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `data/models/`    | Folder of trained models. |
| `data/img/sudoku/`    | example puzzles that are tested with. |
| `data/NumberDetection/`    | Dataset used for model training |
| `src/`            | Core implementation scripts or modules.                           |
| `README.md`       | This file.                                                         |


## ✅ TODO List

- [x] Create CSV input
- [x] Design Datastructure
- [x] Implement Puzzle Detection and Transform
- [x] Create Dataset of individual sudoku cell images
- [x] Train initial model
- [x] Establish pipeline from puzzle image > transform > detect numbers > save in datastructure
- [x] Once pipeline is completed, improve documentation
- [ ] Research Sudoku Rules and rank into easy, moderate or hard
- [ ] Implement easy rules
- [ ] Implement moderate rules
- [ ] Implement Hard rules
- [ ] Implement GUI with image selector
- [ ] Train new dataset that detects handwritten digits.
- [ ] Expand from puzzle solving to giving hints and providing individual steps

## 🚀 Getting Started

At this stage the best way to run is:
<br><br>
`$ python src/vis_reader.py` 
<br><br>
Which may still have some sanity prints enabled but will load the example puzzle using the functions `puzzle_build_viz()` and the `sudoku_puzzle` method `.print_puzzle()`