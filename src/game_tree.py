import numpy as np
import math
import connectFour
import copy

class SearchTree:
    def __init__(self, exploration_factor):
        self.__gamestates = {}
        self.exploration_factor = exploration_factor
        # self.neural_net = neural_net

    # gamestate is a 2d array
    # q is vector of the expected reward for taking action a from gamestate
    # n is vector the number of times we took action a from gamestates across simulations
    # p is vector the initial estimate of taking an action from state s by neural network
    # to play is a number indicating 1 or -1 for player who went first or second
    # num_win is number of times this gamestate has lead to a num_win
    # num_play is total number of times this gamestate has been visited

    def add(self, gamestate, q, n, p, to_play):
        self.__gamestates[tuple(gamestate)] = [q, n, p, to_play]

    def search(self, gamestate):
        return self.__gamestates[tuple(gamestate)]

    def rollout(self, gamestate, to_play):
        while(connectFour.checkWinner(gamestate) == 2):
            connectFour.play(gamestate, to_play, connectFour.random_valid(gamestate))
            to_play*=-1
        return connectFour.checkWinner(gamestate)*to_play

    def self_play (self, depth):
        board = np.zeros((6,7))
        to_play = 1
        while(connectFour.checkWinner(board) == 2):
            connectFour.print_board(board)
            if tuple(map(tuple,board)) not in self.__gamestates:
                # result = self.neural_net.feedforward(gamestate)
                result = np.random.randn(7)
                result = result/np.sum(result)
                v_prime = result
                self.__gamestates[tuple(map(tuple,board))] = np.array([[0]*7, [0]*7, v_prime, to_play])
            #for each move, build the tree from root gamestate
            for i in range(depth):
                self.select(copy.copy(board))
            connectFour.play(board, to_play, self.__gamestates[tuple(map(tuple,board))][1].index(max(self.__gamestates[tuple(map(tuple,board))][1])))
            to_play*=-1
        connectFour.print_board(board)
        training_set = []
        for key, value in self.__gamestates.viewitems():
            if np.sum(value[1]) != 0:
                training_set.append((np.array(key), value[1]/np.sum(value[1]), connectFour.checkWinner(board)))
        return training_set

    def select(self, gamestate):
        # if terminal node
        win = connectFour.checkWinner(gamestate)
        if win != 2:
            return -1*win

        tupled_gamestate = tuple(map(tuple, gamestate))
        # get q, n, p, and toplay values
        stats = self.__gamestates[tupled_gamestate]

        #shuffle valid moves randomly
        valid_moves = []
        for i in range(7):
            if connectFour.check_valid(gamestate, i):
                valid_moves.append(i)
        np.random.shuffle(valid_moves)

        move = -1
        #calculate upper confidence bound, to pick a move
        max_ucb = -1*float("inf")
        for i in range(0, len(valid_moves)):
            if max_ucb <= stats[0][valid_moves[i]] + self.exploration_factor*stats[2][valid_moves[i]]*math.sqrt(np.sum(stats[1]))/(1+stats[1][valid_moves[i]]):
                move = valid_moves[i]

        #generate new state
        new_state = connectFour.play(gamestate, stats[3], move)
        tupled_new_state = tuple(map(tuple,new_state))

        # if the new gamestate already exists, update q value
        if tupled_new_state in self.__gamestates:
            win = self.select(new_state)
            self.__gamestates[tupled_gamestate][0][move] = (self.__gamestates[tupled_gamestate][0][move]*self.__gamestates[tupled_gamestate][1][move] + win)/(self.__gamestates[tupled_gamestate][1][move] +1)
            self.__gamestates[tupled_gamestate][1][move] += 1
            return -1*win

        # result = self.neural_net.feedforward(gamestate)
        # v_prime = result [:-1]
        result = np.random.randn(7)
        result = result/np.sum(result)
        v_prime = result
        new_stats = np.array([[0]*7, [0]*7, v_prime, -1*stats[3]])
        self.__gamestates[tupled_new_state] = new_stats
        return -1*self.rollout(gamestate, -1*stats[3])



tree = SearchTree(0.5)
training_set = tree.self_play(5)
print np.shape(training_set)
