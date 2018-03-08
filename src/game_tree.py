import numpy as np
import math
import connectFour
import copy
import time
from neuralnets import ConvolutionalNet

class SearchTree:
    def __init__(self, exploration_factor, neural_net):
        self.__gamestates = {}
        self.exploration_factor = exploration_factor
        self.neural_net = neural_net

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
        #print "Rollout complete, winner is: " + str(connectFour.checkWinner(gamestate))
        return connectFour.checkWinner(gamestate)*to_play

    def self_play (self, depth):
        board = np.zeros((6,7))
        to_play = 1
        while(connectFour.checkWinner(board) == 2):
            # connectFour.print_board(board)
            if tuple(map(tuple,board)) not in self.__gamestates:
                result = self.neural_net.feedforward(board)
                v_prime = result[:-1]
                self.__gamestates[tuple(map(tuple,board))] = np.array([[0.0]*7, [0.0]*7, v_prime, to_play])
            #for each move, build the tree from root gamestate
            for i in range(depth):
                self.select(copy.copy(board))
            connectFour.play(board, to_play, self.__gamestates[tuple(map(tuple,board))][1].index(max(self.__gamestates[tuple(map(tuple,board))][1])))
            to_play*=-1
        # connectFour.print_board(board)
        training_set = []
        summ = 0
        for key, value in self.__gamestates.viewitems():
            if np.sum(value[1]) > 15:
                summ+= np.sum(self.__gamestates[key][1])
                training_set.append((np.array([np.array(key)]), np.append(value[1]/np.sum(value[1]),(np.sum(value[0])/np.sum(value[1])))))

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
            if max_ucb <= stats[0][valid_moves[i]]*(1-self.exploration_factor) + self.exploration_factor*stats[2][valid_moves[i]]*math.sqrt(np.sum(stats[1]))/(1+stats[1][valid_moves[i]]):
                move = valid_moves[i]

        #generate new state
        new_state = connectFour.play(gamestate, stats[3], move)
        #connectFour.print_board(new_state)

        tupled_new_state = tuple(map(tuple,new_state))

        # if the new gamestate already exists, update q value
        if tupled_new_state in self.__gamestates:
            win = self.select(new_state)
            self.__gamestates[tupled_gamestate][0][move] = (self.__gamestates[tupled_gamestate][0][move]*self.__gamestates[tupled_gamestate][1][move] + win)/(self.__gamestates[tupled_gamestate][1][move] +1)
            self.__gamestates[tupled_gamestate][1][move] += 1
            #print self.__gamestates[tupled_gamestate]
            return -1*win
        else:
            #print "LEAF NODE, starting rollout"
            result = self.neural_net.feedforward(gamestate)
            v_prime = result [:-1]
            new_stats = np.array([[0.0]*7, [0.0]*7, v_prime, -1*stats[3]])
            self.__gamestates[tupled_new_state] = new_stats
            win = self.rollout(gamestate, -1*stats[3])


            # print np.sum(self.__gamestates[tupled_gamestate][1])
            return -1*self.rollout(gamestate, -1*stats[3])

cnn = ConvolutionalNet((1,6,7))
cnn.addlayer("conv", None, (4,3,3))
cnn.addlayer("conv", None, (4,3,3))
cnn.addlayer("dense", 20)
cnn.addlayer("out")

for hyperepoch in range(100):
    print "Hyper Epoch: " + str(hyperepoch)
    cnn_new = copy.deepcopy(cnn)
    tree = SearchTree(0.5, cnn_new)
    for i in range(10):
        print "Game: " + str(i)
        tree.self_play(30)
    training_set = tree.self_play(30)
    # print training_set
    # print np.shape(training_set)
    # print training_set[0][1]

    cnn.stochastic_gradient_descent(epochs=10,
                                    step_size=0.03,
                                    mini_batch_size=len(training_set)/10,
                                    training_set=training_set,
                                    is_momentum_based=False,
                                    friction=0.9)

while True:
    board = np.zeros((6,7))
    connectFour.print_board(board)
    while(connectFour.checkWinner(board) ==2):
        move = input("make a move: ")
        if not connectFour.check_valid(board, move):
            continue
        #connectFour.play(board, 1, minimax.pickMove(board, 1, 3, net0))
        connectFour.play(board, 1, move)
        connectFour.print_board(board)
        # raw_input("press")
        print
        if not connectFour.checkWinner(board)==2:
            break
        results = cnn.feedforward(np.array([board]))[:-1]
        connectFour.play(board, -1, np.where(results == max(results)))
        connectFour.print_board(board)
    print ("WINNER:" + str(connectFour.checkWinner(board)))
