# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import time

from pacman import Directions
from game import Agent
from heuristics import *
import random

class RandomAgent(Agent):

    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
       # print(legal)
        #print(state)
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        #print(successors)
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in successors]
       # print(scored)
        # get best choice
        bestScore = min(scored)[0]
       # print(bestScore)
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
       # print(bestActions)
        print("----------------------------------------")
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):

        # get all legal actions for pacman
        lLegalOptions = state.getLegalPacmanActions()
        lOptionsScore = []

        # No of succesor allowed for one option.If win state is not found within the following number,give other options chance to find the win state.
        liSuccessorAllowedForOneOption = 495/len(lLegalOptions)

        for lOption in lLegalOptions:
            #do bfs for each option
            liNoOfSuccesorCounter = int(0)
            lCurrState = state.generatePacmanSuccessor(lOption)
            liFailureChance = int(2000)
            lStatesQueue = []
            lStatesQueue.insert(0, (lCurrState,0))
            liFPath = int(0)

            while not len(lStatesQueue) == 0:                       #while queue is not empty
                lState, liPath  = lStatesQueue.pop()                #pop from queue
                liFPath = liPath

                #check if state wins,loses or runs out of succesors
                if lState is None:
                    break
                elif lState.isWin():
                    liFailureChance = 0
                    break
                elif lState.isLose():
                    liFailureChance = 1000.0
                    liFPath = 999 - liFPath
                    break
                else:
                    liFailureChance = admissibleHeuristic(lState)       #less heuristics means greater chance of goal state

                if liNoOfSuccesorCounter >= liSuccessorAllowedForOneOption:
                    continue

                lFActions = lState.getLegalPacmanActions()                  #get child nodes
                for lFAction in lFActions:
                    lSuccessorState = lState.generatePacmanSuccessor(lFAction)
                    liNoOfSuccesorCounter = liNoOfSuccesorCounter + 1
                    if lSuccessorState is None or liNoOfSuccesorCounter >= liSuccessorAllowedForOneOption:      #run out of successor then break
                        #print("Last " + str(liNoOfSuccesorCounter))
                        break
                    if admissibleHeuristic(lSuccessorState) < admissibleHeuristic(lState):                      #if child is already visited do not add in queue
                        lStatesQueue.insert(0, (lSuccessorState,liPath+1))
            lOptionsScore.append((liFailureChance, liFPath, lOption))

        #print(lOptionsScore)
        lBestScore = min(lOptionsScore)[0]
        #select action with less failure chance
        lBestActions = [ (lChoices[1], lChoices[2]) for lChoices in lOptionsScore if lChoices[0] == lBestScore]
        #if failure chances are equal select the one with lowest path
        #print(lBestActions)
        lBestPathScore = min(lBestActions)[0]

        lFBestAction = [pair[1] for pair in lBestActions if pair[0] == lBestPathScore]

        # return random action from the list of the best actions
        return random.choice(lFBestAction)


class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):

        # get all legal actions for pacman
        lLegalOptions = state.getLegalPacmanActions()
        lOptionsScore = []

        # No of succesor allowed for one option.If win state is not found within the following number,give other options chance to find the win state.
        liSuccessorAllowedForOneOption = 495/len(lLegalOptions)

        for lOption in lLegalOptions:
            liNoOfSuccesorCounter = int(0)
            lCurrState = state.generatePacmanSuccessor(lOption)
            liFailureChance = int(2000)
            lStatesStack = []
            liFPath = int(0)
            lStatesStack.append((lCurrState,0))

            while not len(lStatesStack) == 0:                                       #while stack is not empty
                lState, liPath = lStatesStack.pop()                                  #pop from stack
                liFPath = liPath

                if lState is None:
                    break
                elif lState.isWin():
                    liFailureChance = 0
                    break
                elif lState.isLose():
                    liFailureChance = 1000.0
                    liFPath = 999 - liFPath
                    break
                else:
                    liFailureChance = admissibleHeuristic(lState)

                if liNoOfSuccesorCounter >= liSuccessorAllowedForOneOption:
                    continue

                lFActions = lState.getLegalPacmanActions()
                for lFAction in lFActions:                                                                          #get child nodes
                    lSuccessorState = lState.generatePacmanSuccessor(lFAction)
                    liNoOfSuccesorCounter = liNoOfSuccesorCounter + 1
                    if lSuccessorState is None or liNoOfSuccesorCounter >= liSuccessorAllowedForOneOption:
                        #print("Last " + str(liNoOfSuccesorCounter))
                        break
                    if admissibleHeuristic(lSuccessorState) < admissibleHeuristic(lState):                           #if child is already visited do not add in stack
                        lStatesStack.append((lSuccessorState, liPath+1))

            lOptionsScore.append((liFailureChance, liFPath, lOption))

        lBestScore = min(lOptionsScore)[0]

        lBestActions = [ (lChoices[1], lChoices[2]) for lChoices in lOptionsScore if lChoices[0] == lBestScore]

        lBestPathScore = min(lBestActions)[0]

        lFBestAction = [pair[1] for pair in lBestActions if pair[0] == lBestPathScore]

        # return random action from the list of the best actions
        return random.choice(lFBestAction)



class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):

        # get all legal actions for pacman
        lLegalOptions = state.getLegalPacmanActions()
        lOptionsScore = []

        # No of succesor allowed for one option.If win state is not found within the following number,give other options chance to find the win state.
        liSuccessorAllowedForOneOption = 495/len(lLegalOptions)

        for lOption in lLegalOptions:
            liNoOfSuccesorCounter = int(0)
            lCurrState = state.generatePacmanSuccessor(lOption)
            liFailureChance = int(2000)
            lStatesPriorityQueue = []
            lStatesPriorityQueue.append(((lCurrState, 0),0))
            liFPath = int(0)

            while not len(lStatesPriorityQueue) == 0:                                               #while priorityqueue is not empty
                # and liNoOfSuccesorCounter < liSuccessorAllowedForOneOption:
                lStatesPriorityQueue.sort(key=lambda x : x[1], reverse=True)                        #sort the list so it becomes priority queue,reverse to get the lowest value

                (lState,lPath), lPValue = lStatesPriorityQueue.pop()
                liFPath = lPath

                if lState is None:
                    break
                elif lState.isWin():
                    liFailureChance = 0
                    break
                elif lState.isLose():
                    liFailureChance = 1000.0
                    liFPath = 999 - liFPath
                    break
                else:
                    liFailureChance = admissibleHeuristic(lState)

                if liNoOfSuccesorCounter >= liSuccessorAllowedForOneOption:
                    continue


                lFActions = lState.getLegalPacmanActions()
                for lFAction in lFActions:
                    lSuccessorState = lState.generatePacmanSuccessor(lFAction)
                    liNoOfSuccesorCounter = liNoOfSuccesorCounter + 1
                    if lSuccessorState is None or liNoOfSuccesorCounter >= liSuccessorAllowedForOneOption:
                        # print("Last " + str(liNoOfSuccesorCounter))
                        break

                    #print(admissibleHeuristic(lSuccessorState))
                    if admissibleHeuristic(lSuccessorState) < admissibleHeuristic(lState):                                          #if child is already visited do not add in priority queue
                        lStatesPriorityQueue.append( ((lSuccessorState, lPath+1), ( admissibleHeuristic(lSuccessorState) + lPath + 1 ) ) )   #f(n) = cost of the path since the start node + heuristics

            lOptionsScore.append((liFailureChance, liFPath, lOption))

        #print(lOptionsScore)

        lBestScore = min(lOptionsScore)[0]

        lBestActions = [(lChoices[1], lChoices[2]) for lChoices in lOptionsScore if lChoices[0] == lBestScore]

        lBestPathScore = min(lBestActions)[0]

        lFBestAction = [pair[1] for pair in lBestActions if pair[0] == lBestPathScore]

        # return random action from the list of the best actions
        return random.choice(lFBestAction)

