Othello (also known as Reversi) is a strategy game played on an 8 x 8 gameboard. The rules to the game and a brief explanation of strategy issues are available on the Wikipedia page for Reversi(https://en.wikipedia.org/wiki/Reversi). 

The shared-memory parallel program is written in Cilk Plus that enables the computer to play the game. The computer is implemented as a parallel function that plays Othello by searching n moves ahead to select “the best board position” for its move.

Input specification:

The program prompt for the user to enter an 'h' or 'c' to specify if player 1 is a human or computer player.
If player 1 is a computer player, it should prompt for an integer between 1 and 60 that specifies its search depth.
The program should prompt for the user to enter an 'h' or 'c' to specify if player 2 is a human or computer player.
If player 2 is a computer player, it should prompt for an integer between 1 and 60 that specifies the search depth (the number of moves ahead the computer should explore when choosing its move).

Output specification:

Given an input file as specified above for two computer players, the program should run to completion. When the program begins, it should print the initial game board configuration. After each move, the program should print the 'row,column' position of the move, specify which disks were flipped, indicate how many disks were flipped, and reprint the game board. At the end of the game, the program should print out the final game board, the total number of disks for each player, and announce the winner. 
