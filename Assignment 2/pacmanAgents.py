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


from pacman import Directions
from game import Agent
from heuristics import *
import random
import math

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class HillClimberAgent(Agent):

    def registerInitialState(self, state):
        self.mActionList = []
        for i in range(0, 5):
            self.mActionList.append(Directions.STOP)
        return

    def getNewStateScore(self, pInitialState):
            lTempState = pInitialState

            for i in range(0, len(self.mActionList)):
                if lTempState.isWin() + lTempState.isLose() == 0:
                    lTempState = lTempState.generatePacmanSuccessor(self.mActionList[i]);
                    if lTempState is None:
                        break
                else:
                    break

            if lTempState is None:
                return gameEvaluation(pInitialState, pInitialState), False
            else:
                return gameEvaluation(pInitialState, lTempState), True

    #generate Actionlist from Available actions  or Given List depending on probability
    def generateNewActionList(self, pBestActionList, pAvailableActions, pProbability):
        for i in range(0, len(self.mActionList)):
            randomNumber = random.randint(1, 100)
            if(randomNumber <= pProbability):
                self.mActionList[i] = pAvailableActions[random.randint(0, len(pAvailableActions) - 1)]
            else:
                self.mActionList[i] = pBestActionList[i]

    def getAction(self, state):

        lAllPossibleActions = state.getAllPossibleActions()
        lOriginalState = state
        lSuccessorNotEmpty = True
        lBestScore = -10001
        lBestActionList = list(self.mActionList)

        while lSuccessorNotEmpty:
            if lBestScore == -10001:
                self.generateNewActionList(lBestActionList, lAllPossibleActions, 100)
            else:
                self.generateNewActionList(lBestActionList, lAllPossibleActions, 50)

            lNewScore, lSuccessorNotEmpty = self.getNewStateScore(lOriginalState)

            # Hill Climb
            if lNewScore > lBestScore:
                lBestActionList = list(self.mActionList)
                lBestScore = lNewScore

            #This step ensures that the bestaction isn't always the first best action sequence encounter but
            # a random one from available best action sequence
            if lNewScore == lBestScore:
                if random.randint(0, 1) == 1:
                    lBestActionList = list(self.mActionList)
                    lBestScore = lNewScore

        return lBestActionList[0]

class GeneticAgent(Agent):
    
    def registerInitialState(self, state):
        self.mPopulationList = []
        for i in range(0,8):
            lActionList = []
            for j in range(0, 5):
                lActionList.append(Directions.STOP)
            self.mPopulationList.append((lActionList, -10001))
        return

    def getNewStateScore(self, pInitialState, pActionList):
            lTempState = pInitialState

            for i in range(0, len(pActionList)):
                if lTempState.isWin() + lTempState.isLose() == 0:
                    lTempState = lTempState.generatePacmanSuccessor(pActionList[i])
                    if lTempState is None:
                        break
                else:
                    break

            if lTempState is None:
                return gameEvaluation(pInitialState, pInitialState), False
            else:
                return gameEvaluation(pInitialState, lTempState), True

    def sortPopulation(self, pPopulationList):
        pPopulationList.sort(key=lambda x : x[1], reverse=True)

    def crossover(self, pFirstParent, pSecondParent):
        lChild = []
        for j in range(0, 5):
            if random.randint(1, 100) <= 50:
                lChild.append(pFirstParent[j])
            else:
                lChild.append(pSecondParent[j])
        return lChild

    def mutate(self, pChild, pPossibleActions):
        for j in range(0, 5):
            pChild[j] = pPossibleActions[random.randint(0, len(pPossibleActions) - 1)]
        return pChild

    def getParentIndex(self,pParentProbabilty):
        # 1st parent=0-7, 2p=8-14, 3p=15-20, 4p=21-25, 5p=26-29, 6p=30-32, 7p=33-34, 8p=35
        if (pParentProbabilty == 35):
            return 7
        if (pParentProbabilty >= 33 and pParentProbabilty <= 34):
            return 6
        if (pParentProbabilty >= 30 and pParentProbabilty <= 32):
            return 5
        if (pParentProbabilty >= 26 and pParentProbabilty <= 29):
            return 4
        if (pParentProbabilty >= 21 and pParentProbabilty <= 25):
            return 3
        if (pParentProbabilty >= 15 and pParentProbabilty <= 20):
            return 2
        if (pParentProbabilty >= 8 and pParentProbabilty <= 14):
            return 1
        return 0

    # GetAction Function: Called with every frame
    def getAction(self, state):

        possible = state.getAllPossibleActions()
        lSuccessorNotEmpty = True
        lOriginalState = state
        liSuccessorLimitCounter = 0

        #initialize population
        for i in range(0,8):
            lActionList, lNewScore = self.mPopulationList[i]
            for j in range(0, 5):
                lActionList[j] = possible[random.randint(0, len(possible)-1)]
                lNewScore, lSuccessorNotEmpty = self.getNewStateScore(lOriginalState, lActionList)
                liSuccessorLimitCounter = liSuccessorLimitCounter + 1
            self.mPopulationList[i] = (lActionList, lNewScore)

        self.sortPopulation(self.mPopulationList)

        while lSuccessorNotEmpty and liSuccessorLimitCounter + 40 < 330:

            lNewPopulation = []

            for i in range(0, 4):
                lFirstParentIndex = self.getParentIndex(random.randint(0, 35))
                lSecondParentIndex = self.getParentIndex(random.randint(0, 35))

                lFirstParent, lNewScore = self.mPopulationList[lFirstParentIndex]
                lSecondParent, lNewScore = self.mPopulationList[lSecondParentIndex]

                #crossover
                lFirstChild = lFirstParent
                lSecondChild = lSecondParent
                if( random.randint(1, 100) <= 70 ):
                    lFirstChild = self.crossover(lFirstParent, lSecondParent)
                    lSecondChild = self.crossover(lFirstParent, lSecondParent)

                #mutation
                if (random.randint(1, 100) <= 10):
                    lFirstChild = self.mutate(lFirstChild, possible)

                if (random.randint(1, 100) <= 10):
                    lSecondChild = self.mutate(lSecondChild, possible)

                lNewPopulation.append(lFirstChild)
                lNewPopulation.append(lSecondChild)

            # add new population to population
            for i in range(0, 8):
                lNewScore, lSuccessorNotEmpty = self.getNewStateScore(lOriginalState, lNewPopulation[i])
                liSuccessorLimitCounter = liSuccessorLimitCounter + 5
                self.mPopulationList[i] = (lNewPopulation[i], lNewScore)

            self.sortPopulation(self.mPopulationList)

        # return the best population's first direction
        return self.mPopulationList[0][0][0]

class MCTSAgent(Agent):
   
    def registerInitialState(self, state):
        self.mNodeList = []
        self.mTotalNodesVisited = 0
        self.lSuccessorNotEmpty = True
        return

    def selection(self, pState, pPID):

        lBestscore = -10001
        lSelectedNode = -1
        lIsSelectedNodeExpanded = 0
        lSelectedNodeDirection = Directions.STOP
        lC = 1

        for lNode in self.mNodeList:
            if lNode[2] == pPID:
                lNj = lNode[4]
                #TODO * 10
                lExploitation = ( lNode[3]/lNj ) * 10
                lExploration = lC * math.sqrt( (2 * math.log(self.mTotalNodesVisited))/lNj )
                lFScore = lExploitation + lExploration

                if lFScore > lBestscore:
                    lBestscore = lFScore
                    lSelectedNode = lNode[0]
                    lSelectedNodeDirection = lNode[1]
                    lIsSelectedNodeExpanded = lNode[5]

        lSelectedNodeState = pState.generatePacmanSuccessor(lSelectedNodeDirection)

        if lSelectedNodeState is None:
            self.lSuccessorNotEmpty = False
            return pState, pPID, 0

        return lSelectedNodeState, lSelectedNode, lIsSelectedNodeExpanded

    #expand,simulate,backpropogate
    def ESB(self, pState, pPID):

        lLegalOptions = pState.getLegalPacmanActions()
        counter = 0
        for lOption in lLegalOptions:
            counter = counter + 1

            #expansion
            lChildState = pState.generatePacmanSuccessor(lOption)
            if lChildState is None:
                self.lSuccessorNotEmpty = False
                return

            #simulation
            lScore, self.lSuccessorNotEmpty = self.simulate(lChildState)
            if self.lSuccessorNotEmpty == False:
                return

            #backpropogate
            self.backpropogate(lScore, pPID)

        # Id, Direction, ParentId, Score, TimesNodeVisted, NofChildren
            self.mNodeList.append((str(pPID) + str(counter), lOption, pPID, lScore, 1, 0))
            self.mTotalNodesVisited = self.mTotalNodesVisited + 1

        self.updateParent(pPID, len(lLegalOptions))
        return

    def simulate(self, pInitialState):

        lTempState = pInitialState

        for i in range(0, 5):
            if lTempState.isWin() + lTempState.isLose() == 0:
                lLegalActions = lTempState.getLegalPacmanActions()
                lTempState = lTempState.generatePacmanSuccessor(lLegalActions[random.randint(0, len(lLegalActions)-1)])
                if lTempState is None:
                    break
            else:
                break

        if lTempState is None:
            return gameEvaluation(pInitialState, pInitialState), False
        else:
            return gameEvaluation(pInitialState, lTempState), True


    def backpropogate(self, pScore, pPID):

        if pPID == 0:
            return

        for i, lNode in enumerate(self.mNodeList):
            if lNode[0] == pPID:
                lPID = lNode[2]
                self.mNodeList[i] = (lNode[0], lNode[1], lNode[2], lNode[3] + pScore, lNode[4] + 1, lNode[5])
                self.backpropogate(pScore, lPID)
                break

        return

    def updateParent(self, pPID, pNoOfChildren):

        if pPID == 0:
            return

        for i, lNode in enumerate(self.mNodeList):
            if lNode[0] == pPID:
                self.mNodeList[i] = (lNode[0], lNode[1], lNode[2], lNode[3], lNode[4], pNoOfChildren)
                break

        return

    def getBestChildDirection(self):

        lBestScore = -10001
        lBestDirection = Directions.STOP
        for lNode in self.mNodeList:
            lNID = lNode[0]
            if len(str(lNID)) == 2:
                lScore = lNode[3] / lNode[4]
                if lScore > lBestScore:
                    lBestScore = lScore
                    lBestDirection = lNode[1]

        return lBestDirection

    def getAction(self, state):

        self.mNodeList = []
        self.mTotalNodesVisited = 0
        self.lSuccessorNotEmpty = True

        currentNodeId = 0
        currentState = state

        while self.lSuccessorNotEmpty:

            if currentState.isWin() + currentState.isLose() == 0:
                self.ESB(currentState, currentNodeId)

            if self.lSuccessorNotEmpty == False:
                break

            currentNodeId = 0
            currentState = state
            
            while True:
                if currentState.isWin() + currentState.isLose() == 1:
                    currentNodeId = 0
                    currentState = state
                currentState, currentNodeId, currentNodeExpanded = self.selection(currentState, currentNodeId)
                if currentNodeExpanded == 0:
                    break

        return self.getBestChildDirection()
