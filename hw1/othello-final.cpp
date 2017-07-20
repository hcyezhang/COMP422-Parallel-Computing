#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cilk/cilk.h>
#include <cilk/reducer_max.h>
#include <cilk/reducer_min.h>
#include <cilk/cilk_api.h>

#define BIT 0x1

#define X_BLACK 0
#define O_WHITE 1
#define OTHERCOLOR(c) (1-(c))
/* 
	represent game board squares as a 64-bit unsigned integer.
	these macros index from a row,column position on the board
	to a position and bit in a game board bitvector
*/
#define BOARD_BIT_INDEX(row,col) ((8 - (row)) * 8 + (8 - (col)))
#define BOARD_BIT(row,col) (0x1LL << BOARD_BIT_INDEX(row,col))
#define MOVE_TO_BOARD_BIT(m) BOARD_BIT(m.row, m.col)

/* all of the bits in the row 8 */
#define ROW8 \
  (BOARD_BIT(8,1) | BOARD_BIT(8,2) | BOARD_BIT(8,3) | BOARD_BIT(8,4) |	\
   BOARD_BIT(8,5) | BOARD_BIT(8,6) | BOARD_BIT(8,7) | BOARD_BIT(8,8))
			  
/* all of the bits in column 8 */
#define COL8 \
  (BOARD_BIT(1,8) | BOARD_BIT(2,8) | BOARD_BIT(3,8) | BOARD_BIT(4,8) |	\
   BOARD_BIT(5,8) | BOARD_BIT(6,8) | BOARD_BIT(7,8) | BOARD_BIT(8,8))

/* all of the bits in column 1 */
#define COL1 (COL8 << 7)

#define IS_MOVE_OFF_BOARD(m) (m.row < 1 || m.row > 8 || m.col < 1 || m.col > 8)
#define IS_DIAGONAL_MOVE(m) (m.row != 0 && m.col != 0)
#define MOVE_OFFSET_TO_BIT_OFFSET(m) (m.row * 8 + m.col)

typedef unsigned long long ull;

/* 
	game board represented as a pair of bit vectors: 
	- one for x_black disks on the board
	- one for o_white disks on the board
*/
typedef struct { ull disks[2]; } Board;

typedef struct { int row; int col; } Move;

typedef struct { int score; Move nextmove; } Action;

//Adding global variables to support minimax implementation
#define MAX_SCORE 1000
#define MIN_SCORE -1000

Move no_further_move = {-1,-1};
Action opt_max_action = {MIN_SCORE, no_further_move};
Action opt_min_action = {MAX_SCORE, no_further_move};

struct cmpmax{
    bool operator() (const Action& a, const Action& b) const {
        return a.score < b.score;
    }
};   
struct cmpmin{
    bool operator() (const Action& a, const Action& b) const {
        return a.score > b.score;
    }
}; 

Board start = { 
	BOARD_BIT(4,5) | BOARD_BIT(5,4) /* X_BLACK */, 
	BOARD_BIT(4,4) | BOARD_BIT(5,5) /* O_WHITE */
};
 
Move offsets[] = {
  {0,1}		/* right */,		{0,-1}		/* left */, 
  {-1,0}	/* up */,		{1,0}		/* down */, 
  {-1,-1}	/* up-left */,		{-1,1}		/* up-right */, 
  {1,1}		/* down-right */,	{1,-1}		/* down-left */
};

int noffsets = sizeof(offsets)/sizeof(Move);
char diskcolor[] = { '.', 'X', 'O', 'I' };


void PrintDisk(int x_black, int o_white)
{
  printf(" %c", diskcolor[x_black + (o_white << 1)]);
}

void PrintBoardRow(int x_black, int o_white, int disks)
{
  if (disks > 1) {
    PrintBoardRow(x_black >> 1, o_white >> 1, disks - 1);
  }
  PrintDisk(x_black & BIT, o_white & BIT);
}

void PrintBoardRows(ull x_black, ull o_white, int rowsleft)
{
  if (rowsleft > 1) {
    PrintBoardRows(x_black >> 8, o_white >> 8, rowsleft - 1);
  }
  printf("%d", rowsleft);
  PrintBoardRow((int)(x_black & ROW8),  (int) (o_white & ROW8), 8);
  printf("\n");
}

void PrintBoard(Board b)
{
  printf("  1 2 3 4 5 6 7 8\n");
  PrintBoardRows(b.disks[X_BLACK], b.disks[O_WHITE], 8);
}

/* 
	place a disk of color at the position specified by m.row and m,col,
	flipping the opponents disk there (if any) 
*/
void PlaceOrFlip(Move m, Board *b, int color) 
{
  ull bit = MOVE_TO_BOARD_BIT(m);
  b->disks[color] |= bit;
  b->disks[OTHERCOLOR(color)] &= ~bit;
}

/* 
	try to flip disks along a direction specified by a move offset.
	the return code is 0 if no flips were done.
	the return value is 1 + the number of flips otherwise.
*/
int TryFlips(Move m, Move offset, Board *b, int color, int verbose, int domove)
{
  Move next;
  next.row = m.row + offset.row;
  next.col = m.col + offset.col;

  if (!IS_MOVE_OFF_BOARD(next)) {
    ull nextbit = MOVE_TO_BOARD_BIT(next);
    if (nextbit & b->disks[OTHERCOLOR(color)]) {
      int nflips = TryFlips(next, offset, b, color, verbose, domove);
      if (nflips) {
	if (verbose) printf("flipping disk at %d,%d\n", next.row, next.col);
	if (domove) PlaceOrFlip(next,b,color);
	return nflips + 1;
      }
    } else if (nextbit & b->disks[color]) return 1;
  }
  return 0;
} 

int FlipDisks(Move m, Board *b, int color, int verbose, int domove)
{
  int i;
  int nflips = 0;
	
  /* try flipping disks along each of the 8 directions */
  for(i=0;i<noffsets;i++) {
    int flipresult = TryFlips(m,offsets[i], b, color, verbose, domove);
    nflips += (flipresult > 0) ? flipresult - 1 : 0;
  }
  return nflips;
}

void ReadMove(int color, Board *b)
{
  Move m;
  ull movebit;
  for(;;) {
    printf("Enter %c's move as 'row,col': ", diskcolor[color+1]);
    scanf("%d,%d",&m.row,&m.col);
		
    /* if move is not on the board, move again */
    if (IS_MOVE_OFF_BOARD(m)) {
      printf("Illegal move: row and column must both be between 1 and 8\n");
      PrintBoard(*b);
      continue;
    }
    movebit = MOVE_TO_BOARD_BIT(m);
		
    /* if board position occupied, move again */
    if (movebit & (b->disks[X_BLACK] | b->disks[O_WHITE])) {
      printf("Illegal move: board position already occupied.\n");
      PrintBoard(*b);
      continue;
    }
		
    /* if no disks have been flipped */ 
    {
      int nflips = FlipDisks(m, b,color, 1, 1);
      if (nflips == 0) {
	printf("Illegal move: no disks flipped\n");
	PrintBoard(*b);
	continue;
      }
      PlaceOrFlip(m, b, color);
      printf("You flipped %d disks\n", nflips);
      PrintBoard(*b);
    }
    break;
  }
}

/*
	return the set of board positions adjacent to an opponent's
	disk that are empty. these represent a candidate set of 
	positions for a move by color.
*/
Board NeighborMoves(Board b, int color)
{
  int i;
  Board neighbors = {0,0};
  for (i = 0;i < noffsets; i++) {
    ull colmask = (offsets[i].col != 0) ? 
      ((offsets[i].col > 0) ? COL1 : COL8) : 0;
    int offset = MOVE_OFFSET_TO_BIT_OFFSET(offsets[i]);

    if (offset > 0) {
      neighbors.disks[color] |= 
	(b.disks[OTHERCOLOR(color)] >> offset) & ~colmask;
    } else {
      neighbors.disks[color] |= 
	(b.disks[OTHERCOLOR(color)] << -offset) & ~colmask;
    }
  }
  neighbors.disks[color] &= ~(b.disks[X_BLACK] | b.disks[O_WHITE]);
  return neighbors;
}

/*
	return the set of board positions that represent legal
	moves for color. this is the set of empty board positions  
	that are adjacent to an opponent's disk where placing a
	disk of color will cause one or more of the opponent's
	disks to be flipped.
*/
int EnumerateLegalMoves(Board b, int color, Board *legal_moves)
{
  static Board no_legal_moves = {0,0};
  Board neighbors = NeighborMoves(b, color);
  ull my_neighbor_moves = neighbors.disks[color];
  int row;
  int col;
	
  int num_moves = 0;
  *legal_moves = no_legal_moves;
	
  for(row=8; row >=1; row--) {
    ull thisrow = my_neighbor_moves & ROW8;
    for(col=8; thisrow && (col >= 1); col--) {
      if (thisrow & COL8) {
	Move m = { row, col };
	if (FlipDisks(m, &b, color, 0, 0) > 0) {
	  legal_moves->disks[color] |= BOARD_BIT(row,col);
	  num_moves++;
	}
      }
      thisrow >>= 1;
    }
    my_neighbor_moves >>= 8;
  }
  return num_moves;
}

int HumanTurn(Board *b, int color)
{
  Board legal_moves;
  int num_moves = EnumerateLegalMoves(*b, color, &legal_moves);
  if (num_moves > 0) {
    ReadMove(color, b);
    return 1;
  } else return 0;
}

int CountBitsOnBoard(Board *, int);
//Implement the AI
Action minMax (Board b, int depth, int color)
{
    Action action;
    action.nextmove.row = -1;
    action.nextmove.col = -1;
    if (depth == 0){
        action.score = CountBitsOnBoard(&b, color) - CountBitsOnBoard(&b, OTHERCOLOR(color));
        return action; 
    } 
        Board myBoard = NeighborMoves(b, color);
        ull my_neighbor_moves = myBoard.disks[color];
        std::vector<Move> candidate_set;//generate a candidate set to hold valid moves
        for(int row=8; row >=1; row--) {
            ull thisrow = my_neighbor_moves & ROW8;
            for(int col=8; thisrow && (col >= 1); col--) {
                if (thisrow & COL8) {
                    Move m = { row, col };
                    Board newBoard = b;
                    if (FlipDisks(m, &newBoard, color, 0, 0) > 0){ 
                        candidate_set.push_back(m);
                    }
                }
                thisrow >>= 1;
            }
            my_neighbor_moves >>= 8;
        }
         
        cilk::reducer_max<Action, cmpmax> max_reducer(opt_max_action);
        int n = candidate_set.size();
#pragma cilk grainsize = std::max(1, 7*n/( depth * __cilkrts_get_nworkers() )) 
        cilk_for(int i = 0; i < candidate_set.size(); i++) {
            Board newBoard = b;
            FlipDisks(candidate_set[i], &newBoard, color,0,1);
            PlaceOrFlip(candidate_set[i], &newBoard, color);
            Action val = minMax(newBoard, depth-1, OTHERCOLOR(color));
            val.nextmove = candidate_set[i];
            val.score = -val.score;//Always negate the score returned by the other player
            max_reducer->calc_max(val);
        }
        //Corner Case when no moves are availale for maxPlayer
        if (candidate_set.size() == 0){
            Board legal_moves;
            Board newBoardOppnt = b;
            //Check if the other player can move 
            if(EnumerateLegalMoves(newBoardOppnt, OTHERCOLOR(color), &legal_moves) > 0){
                Action val = minMax(newBoardOppnt, depth, OTHERCOLOR(color));
                action.score = -val.score;
            }
            else{
            action.score = CountBitsOnBoard(&b, color) - CountBitsOnBoard(&b, OTHERCOLOR(color));
            }    
            return action;
        }       
        
        return max_reducer.get_value();
}

/* wrap the minMax function */
Action ComputerTurn(Board b, int depth, int color)
{
    return minMax(b, depth, color);
    
}

int CountBitsOnBoard(Board *b, int color)
{
  ull bits = b->disks[color];
  int ndisks = 0;
  for (; bits ; ndisks++) {
    bits &= bits - 1; // clear the least significant bit set
  }
  return ndisks;
}

void EndGame(Board b)
{
  int o_score = CountBitsOnBoard(&b,O_WHITE);
  int x_score = CountBitsOnBoard(&b,X_BLACK);
  printf("Game over. \n");
  if (o_score == x_score)  {
    printf("Tie game. Each player has %d disks\n", o_score);
  } else { 
    printf("X has %d disks. O has %d disks. %c wins.\n", x_score, o_score, 
	      (x_score > o_score ? 'X' : 'O'));
  }
}

int main (int argc, const char * argv[]) 
{
  /*Read Input params*/
  
  char player1, player2;
  int depth1, depth2; 
  
  printf("Please specify if player1 %c is human or computer:", diskcolor[X_BLACK+1]);
  scanf(" %c", &player1);
  if(player1 == 'c'){
    printf("Please specify the depth:");
    scanf("%d", &depth1);  
  }
  
  printf("Please specify if player2 %c is human or computer:", diskcolor[O_WHITE+1]);
  scanf(" %c", &player2);
  if(player2 == 'c'){
    printf("Please specify the depth:");
    scanf(" %d", &depth2);
  }
  Board gameboard = start;
  int move_possible_player1, move_possible_player2;
  PrintBoard(gameboard);
  do {
      if(player1 == 'h') move_possible_player1 = HumanTurn(&gameboard, X_BLACK);
      else if(player1 == 'c'){
      Action computer_action = ComputerTurn (gameboard, depth1, X_BLACK);
        if (computer_action.nextmove.row != -1 && computer_action.nextmove.col != -1){
          printf("Computer placed %c at %d, %d, Score %d\n", diskcolor[X_BLACK+1], computer_action.nextmove.row, computer_action.nextmove.col, computer_action.score);
          int nfilps = FlipDisks(computer_action.nextmove, &gameboard, X_BLACK,0,1);
          PlaceOrFlip(computer_action.nextmove, &gameboard, X_BLACK);
          PrintBoard(gameboard);
          move_possible_player1 = 1;
        }
        else move_possible_player1 = 0;
      } 
      if (player2 == 'h') move_possible_player2 = HumanTurn(&gameboard, O_WHITE);
      else if (player2 == 'c'){
        Action computer_action = ComputerTurn(gameboard, depth2, O_WHITE);
        if(computer_action.nextmove.row != -1 && computer_action.nextmove.col!= -1){
            printf("Computer placed %c at %d. %d, Score %d\n", diskcolor[O_WHITE+1], computer_action.nextmove.row, computer_action.nextmove.col, computer_action.score);
            int nflips = FlipDisks(computer_action.nextmove, &gameboard, O_WHITE, 0,1);
            PlaceOrFlip(computer_action.nextmove, &gameboard, O_WHITE); 
            PrintBoard(gameboard);
            move_possible_player2 = 1;
        }
        else move_possible_player2 = 0;
      }  
  } while(move_possible_player1 || move_possible_player2);

  EndGame(gameboard);
	
  return 0;
}
