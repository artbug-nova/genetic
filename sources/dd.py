import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import euclidean

class Robot:
    parts = []
    penaltiesMin = None
    penaltiesMax = None

    def __init__(self, parts):
        self.parts = parts
        self.penaltiesMin = [(p.borderMin) for p in self.parts]
        self.penaltiesMax = [(p.borderMax) for p in self.parts]

    '''
    Получить значение штрафа для данных обобщенных координат
    '''


    def penalty(self, Q, W1=1, W2=1):

        #lambda (t): (200 * exp(-t)) if t > 200 else (400 * exp(-t))
        reduce_to_nil = lambda n: 0 if n > 0 else np.abs(n)

        subtract = np.subtract(Q, self.penaltiesMin)
        np_subtract = np.subtract(self.penaltiesMax, Q)
        return W1 * np.sum(list(map(reduce_to_nil, subtract))) \
               + W2 * np.sum(list(map(reduce_to_nil, np_subtract)))

    '''
    Получить координаты схвата (конечного звена)
    '''

    def getXYZ(self, Q):
        return self.getXYZPair(Q, len(self.parts))[:3]

    '''
    Получить координаты конкретной пары 
    '''

    def getXYZPair(self, Q, pair):

        resultMatrix = np.eye(4, dtype=np.float32)

        for i, p in enumerate(self.parts):

            if i == pair:
                break

            resultMatrix = np.matmul(resultMatrix, p.getMatrix(Q[i]))

        xyz1 = np.matmul(resultMatrix, [[0], [0], [0], [1]])

        return xyz1

    '''
    Массив координат всех пар (для построения графика)
    '''

    def getPairPoints(self, Q):

        result = []

        for i, p in enumerate(self.parts):
            pairXYZ = self.getXYZPair(Q, i)
            result.append([pairXYZ[0], pairXYZ[1], pairXYZ[2]])

        return result




class KinematicPart:
    s = 0
    a = 0
    alpha = 0

    borderMin = 0
    borderMax = 0

    def __init__(self, s, a, alpha, bmin, bmax):
        self.s = s
        self.a = a
        self.alpha = alpha
        self.borderMin = bmin
        self.borderMax = bmax

    def getMatrix(self, q):
        return [
            [np.cos(q), -np.sin(q) * np.cos(self.alpha), np.sin(q) * np.sin(self.alpha), self.a * np.cos(q)],
            [np.sin(q), np.cos(q) * np.cos(self.alpha), -np.cos(q) * np.sin(self.alpha), self.a * np.sin(q)],
            [0, np.sin(self.alpha), np.cos(self.alpha), self.s],
            [0, 0, 0, 1]]


r = np.pi / 180.0

#
Z1 = KinematicPart(300, 0, np.pi / 2, bmin=-185 * r, bmax=185 * r)
Z2 = KinematicPart(0, 250, 0, bmin=50 * r, bmax=270 * r)
Z3 = KinematicPart(0, 160, 0, bmin=-360 * r, bmax=360 * r)
Z4 = KinematicPart(0, 0, np.pi / 2, bmin=40 * r, bmax=40 * r)
Z5 = KinematicPart(0, 104.9, np.pi / 2, bmin=-5 * r, bmax=15 * r)

parts = [Z1, Z2, Z3, Z4, Z5]#, Z6]

RV = Robot(parts)


Q01 = 0 * r
Q12 = 90 * r
Q23 = 270 * r
Q34 = 0 * r
Q45 = 0 * r

target = [[264], [0], [551]]
Q0 = [Q01, Q12, Q23, Q34, Q45]




def loss_function(Q0):
    xyz = RV.getXYZ(Q0)
    penalty = RV.penalty(Q0, 1, 1)
    #return np.sum(np.sqrt((target - xyz) ** 2))# + penalty
    return euclidean(target, xyz) + penalty


class Population:
    def __init__(self, l=8, limits=[(0, 1, 2)], gen=[], use_random=True):
        self.fitness = np.random.rand()
        self.l = l
        self.limits = limits
        self.genotype_len = len(self.limits) * self.l
        if use_random:
            self.genotype = np.random.randint(0, 1, self.genotype_len)
        else:
            self.genotype = np.array(gen)
            self.genotype_len = self.genotype.shape[0]
        self.phenotype = self.decode()

    # Function for decoding genotype
    def decode(self):
        list_phenotype = []
        for i in range(len(self.limits)):
            lower, upper = self.limits[i]
            precission = (upper - lower) / (2 ** self.l - 1)
            _sum = 0
            cnt = 0
            for j in range(i * self.l, i * self.l + self.l):
                _sum += self.genotype[j] * 2 ** cnt
                cnt += 1
            phenotype = _sum * precission + lower
            list_phenotype.append(phenotype)
        return tuple(list_phenotype)


class GeneticAlgorithm:
    def __init__(self, n_generations=10, n_populations=5, prob_crossover=1.0, prob_mutation=0.1, k=3):
        # Here we define simple 2 link arm robot with length l1 = 50 and l2 = 50
        self.robot = RV#RobotArm(links=[50, 50, 50])
        # Initialize GA parameter
        self.n_generations = n_generations
        self.n_populations = n_populations
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.k = k
        # Generate population randomly
        self.populations = []
        for i in range(n_populations):
            # limits equal with joints angle limit in range -pi to pi
            pop = Population(l=16, limits=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])
            self.populations.append(pop)

    # Crossover operation between two parents, result in two children genotype
    def crossover(self, parent_1_idx, parent_2_idx, parent_3_idx, split_idx=[8, 16, 24]):
        genotype_len = self.populations[parent_1_idx].genotype_len
        child_1_genotype = np.hstack((self.populations[parent_1_idx].genotype[0:split_idx[0]], \
                                      self.populations[parent_2_idx].genotype[split_idx[0]:split_idx[1]], \
                                      self.populations[parent_1_idx].genotype[split_idx[1]:split_idx[2]], \
                                      self.populations[parent_2_idx].genotype[split_idx[2]:]))
        child_2_genotype = np.hstack((self.populations[parent_2_idx].genotype[0:split_idx[0]], \
                                      self.populations[parent_1_idx].genotype[split_idx[0]:split_idx[1]], \
                                      self.populations[parent_2_idx].genotype[split_idx[1]:split_idx[2]], \
                                      self.populations[parent_1_idx].genotype[split_idx[2]:]))

        child_3_genotype = np.hstack((self.populations[parent_3_idx].genotype[0:split_idx[0]], \
                                      self.populations[parent_2_idx].genotype[split_idx[0]:split_idx[1]], \
                                      self.populations[parent_3_idx].genotype[split_idx[1]:split_idx[2]], \
                                      self.populations[parent_2_idx].genotype[split_idx[2]:]))
        return child_1_genotype, child_2_genotype, child_3_genotype

    # Mutation operation of children genotype, result in new children genotype
    def mutation(self, child_genotype):
        genotype_len = self.populations[0].genotype_len
        for i in range(genotype_len):
            mutate = np.random.choice([True, False], p=[self.prob_mutation, (1 - self.prob_mutation)])
            if mutate:
                if child_genotype[i] == 0:
                    child_genotype[i] = 1
                else:
                    child_genotype[i] = 0
        return child_genotype

    # Selection operation using tournament selection, result in two best parents from populations
    def tournament_selection(self):
        list_parents_idx = []
        for i in range(3):
            min_fitness = 999.0
            best_parent_idx = -1
            for j in range(self.k):
                accept = False
                while not accept:
                    parent_idx = np.random.choice(np.arange(0, len(self.populations)))
                    if parent_idx not in list_parents_idx:
                        accept = True
                if self.populations[parent_idx].fitness < min_fitness:
                    best_parent_idx = parent_idx
                    min_fitness = self.populations[parent_idx].fitness
            list_parents_idx.append(best_parent_idx)
        return tuple(list_parents_idx)

    # Here evolution process
    def evolution(self):
        for generation in range(self.n_generations):
            print("Generation ", generation)
            # Generate new children
            child_populations = []
            while len(child_populations) < self.n_populations:
                # Select best parent from population
                parent_1_idx, parent_2_idx, parent_3_idx = self.tournament_selection()
                # Crossover operation
                child_1_genotype, child_2_genotype, child_3_genotype = self.crossover(parent_1_idx, parent_2_idx,
                                                                                      parent_3_idx)
                # Mutation operation
                child_1_genotype = self.mutation(child_1_genotype)
                child_2_genotype = self.mutation(child_2_genotype)
                child_3_genotype = self.mutation(child_3_genotype)

                child = Population(l=16, limits=[(-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi)],
                                   gen=child_1_genotype, use_random=True)
                joint_1, joint_2, joint_3 = child.phenotype
                # Get fitness value of new children
                Q0[1] = joint_1
                Q0[2] = joint_2
                Q0[3] = joint_3
                child.fitness = loss_function(Q0)#self.robot.calc_distance_error([joint_1, joint_2, joint_3])
                child_populations.append(child)

                child = Population(l=16, limits=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)],
                                   gen=child_2_genotype, use_random=True)
                joint_1, joint_2, joint_3 = child.phenotype
                Q0[1] = joint_1
                Q0[2] = joint_2
                Q0[3] = joint_3
                child.fitness = loss_function(Q0)
                child_populations.append(child)

                child = Population(l=16, limits=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)],
                                   gen=child_2_genotype, use_random=True)
                joint_1, joint_2, joint_3 = child.phenotype
                Q0[1] = joint_1
                Q0[2] = joint_2
                Q0[3] = joint_3
                child.fitness = loss_function(Q0)
                child_populations.append(child)

                child = Population(l=16, limits=[(-90 * r, 90 * r), (-90 * r, 90 * r), (-90 * r, 90 * r)],
                                   gen=child_2_genotype, use_random=False)
                joint_1, joint_2, joint_3 = child.phenotype
                Q0[1] = joint_1
                Q0[2] = joint_2
                Q0[3] = joint_3
                child.fitness = loss_function(Q0)
                child_populations.append(child)

            # Update current parent with new child and track best population
            best_idx = -1
            best_fitness = 999
            for i in range(self.n_populations):
                self.populations[i] = child_populations[i]
                if self.populations[i].fitness < best_fitness:
                    best_idx = i
                    best_fitness = self.populations[i].fitness
            print("Best Population :", Q0, self.populations[best_idx].fitness)
            print("================================================================================")
            print(RV.getXYZ(Q0))
            print("================================================================================")
        return self.populations[best_idx].phenotype

    def run(self):
        # Here we define target position of robot arm
        self.robot.target_pos = [100, 50]
        # Solving the solution with GA
        joint1, joint2, joint3 = self.evolution()
        # Plot robot configuration
        self.robot.plot([joint1, joint2, joint3])


def main():
    ga = GeneticAlgorithm(n_generations=1000, n_populations=100, k=20)
    ga.run()


if __name__ == "__main__":
    main()








