import random as rng
import numpy as np
from bidict import bidict
import copy as cp
import collections
from functools import total_ordering
import pickle
from datetime import datetime
import matplotlib.pyplot as plt


group_to_number = bidict({"k-15": 0, "k-16": 1, "k-17": 2, "k-18": 3})
subj_to_number = bidict({"AG": 1, "Calc": 2, "DM": 3, "Prog": 4, "Eng": 5, "Empty": 0})

class GeneralOptions:
    def __init__(self, subject, teachers, professors, weeks, max_classes_per_day, groups, small_rooms, lect_rooms):
        self.group_to_subjects = subject
        self.teachers_to_subjects = teachers
        self.groups_to_professors = professors
        self.num_of_weeks = weeks
        self.max_classes_per_day = max_classes_per_day
        self.groups = groups
        self.small_rooms = small_rooms
        self.lect_rooms = lect_rooms
        self.gens = 4
        self.num_working_days = 5
        self.threshold = 10
        self.change_threshold = 20


@total_ordering
class Speciment:
    def __init__(self, ops: GeneralOptions):
        self.ops = ops
        self.schedule_table = np.ndarray((len(self.ops.groups), self.ops.num_of_weeks, self.ops.num_working_days,
                                          self.ops.max_classes_per_day, self.ops.gens), dtype=int)
        self.evaluation = None

    def __eq__(self, other):
        return self.evaluate() == other.evaluate()

    def __lt__(self, other):
        return self.evaluate() < other.evaluate()

    def mutate(self, mutation_percentage):
        for group in self.ops.groups:
            subjects_to_hours = cp.deepcopy(self.ops.group_to_subjects[group])
            for week in range(self.ops.num_of_weeks):
                # days in week
                for day in range(5):
                    for class_num in range(self.ops.max_classes_per_day):
                        rng_num = rng.randint(0, 100)
                        if rng_num < mutation_percentage:
                            self.generate_class(subjects_to_hours, group, week, day, class_num)
        self.evaluation = None

    def evaluate(self):
        if self.evaluation is not None:
            return self.evaluation
        teachers_to_classes = dict()
        rooms_to_classes = dict()
        wasted_class_slot = 0
        collided_teacher_classes = 0
        collided_room_classes = 0
        for group in self.ops.groups:
            for week in range(self.ops.num_of_weeks):
                # days in week
                for day in range(5):
                    day_busieness = np.zeros(self.ops.max_classes_per_day)
                    for class_num in range(self.ops.max_classes_per_day):
                        curr_class = self.schedule_table[group_to_number[group]][week][day][class_num]
                        if curr_class[0] != 0:
                            day_busieness[class_num] = 1
                        if not curr_class[2] in teachers_to_classes:
                            teachers_to_classes[curr_class[2]] = dict()
                        if not (week, day, class_num) in teachers_to_classes[curr_class[2]]:
                            teachers_to_classes[curr_class[2]][(week, day, class_num)] = 1
                        teachers_to_classes[curr_class[2]][(week, day, class_num)] += 1
                        if not curr_class[3] in rooms_to_classes:
                            rooms_to_classes[curr_class[3]] = dict()
                        if not (week, day, class_num) in rooms_to_classes[curr_class[3]]:
                            rooms_to_classes[curr_class[3]][(week, day, class_num)] = 1
                        rooms_to_classes[curr_class[3]][(week, day, class_num)] += 1
                    for i in range(1, self.ops.max_classes_per_day):
                        if day_busieness[i] == 1 and day_busieness[i - 1] == 0:
                            wasted_class_slot += 1

        for teacher, classes_to_num_groups in teachers_to_classes.items():
            if teacher == 0:
                continue
            for num_of_groups in classes_to_num_groups.values():
                collided_teacher_classes += num_of_groups - 1

        for room, classes_to_num_groups in rooms_to_classes.items():
            if room == 0:
                continue
            for num_of_groups in classes_to_num_groups.values():
                collided_room_classes += num_of_groups - 1
        self.evaluation = collided_room_classes*3 + collided_teacher_classes*3 + wasted_class_slot*1
        return self.evaluation

    def _get_subject_list(self, group):
        res = []
        subjects_to_hours = self.ops.group_to_subjects[group]
        for subject_and_hours in subjects_to_hours:
            res.append(subject_and_hours)
        return res

    def generate_class(self, subjects_to_hours, group, week, day, class_num):
        subjects = self._get_subject_list(group)
        subject = rng.choice(subjects)
        cur_cell = self.schedule_table[group_to_number[group]][week][day][class_num]
        if subjects_to_hours[subject] > 0:
            subjects_to_hours[subject] -= 1
        else:
            subject = "Empty"
            cur_cell[0] = subj_to_number[subject]
            cur_cell[1] = 0
            cur_cell[2] = 0
            cur_cell[3] = 0
            return
        is_lecture = rng.randint(0, 1)
        teacher = ""
        room = 0
        if is_lecture == 1:
            room = rng.choice(self.ops.lect_rooms)
            professors = self.ops.groups_to_professors[group]
            for prof in professors:
                possible_subjects = self.ops.teachers_to_subjects[prof]
                if subject in possible_subjects:
                    teacher = prof
        else:
            for cur_teacher, possible_subjects in self.ops.teachers_to_subjects.items():
                if subject in possible_subjects:
                    teacher = cur_teacher
            room = rng.choice(self.ops.small_rooms)
        cur_cell[0] = subj_to_number[subject]
        cur_cell[1] = is_lecture
        cur_cell[2] = int(teacher)
        cur_cell[3] = room

    def get_class(self, group, week, day, class_num):
        return self.schedule_table[group_to_number[group]][week][day][class_num]


class Generation:
    def __init__(self, ops: GeneralOptions):
        self.ops = ops

        self.speciments = []

    def copy(self, speciments):
        copied = self.__class__(self.ops)
        copied.speciments = speciments
        return copied

    def mutate(self, mutate_percentage, gen_mutation_percentage):
        for i in range(len(self.speciments)):
            choice = rng.randint(0, 100)
            if choice < mutate_percentage:
                self.speciments[i].mutate(gen_mutation_percentage)

    def add_speciment(self, speciment: Speciment):
        self.speciments.append(speciment)

    def breed(self, generation_size):
        new_generation = self.__class__(self.ops)
        for i in range(generation_size):
            new_generation.add_speciment(self.cross_parents(rng.choice(self.speciments), rng.choice(self.speciments)))
        return new_generation

    def get_best_speciment(self) -> Speciment:
        return self.select(1, 0).speciments[0]

    def shake(self, percentage):
        for i in range(len(self.speciments)):
            choice = rng.randint(0, 100)
            if choice < percentage:
                self.speciments[i] = self.get_random_speciment()

    def select(self, next_generation_size, lucker_percentage):
        copied_speciments = cp.deepcopy(self.speciments)
        copied_speciments.sort()

        num_of_luckers = (next_generation_size * lucker_percentage) // 100
        num_of_alphas = next_generation_size - num_of_luckers
        next_generation = copied_speciments[:num_of_alphas]
        next_generation.extend(rng.sample(copied_speciments[num_of_alphas:], num_of_luckers))
        return self.copy(next_generation)

    def cross_parents(self, speciment1: Speciment, speciment2: Speciment) -> Speciment:
        offspring = Speciment(self.ops)
        for group in self.ops.groups:
            for week in range(self.ops.num_of_weeks):
                # days in week
                for day in range(5):
                    for class_num in range(self.ops.max_classes_per_day):
                        curr_class1 = speciment1.get_class(group, week, day, class_num)
                        curr_class2 = speciment2.get_class(group, week, day, class_num)
                        curr_class_offspring = offspring.get_class(group, week, day, class_num)
                        for i in range(self.ops.gens):
                            choice = rng.randint(0, 1)
                            if choice == 0:
                                curr_class_offspring[i] = curr_class1[i]
                            else:
                                curr_class_offspring[i] = curr_class2[i]
        return offspring

    def get_random_speciment(self) -> Speciment:
        new_speciment = Speciment(self.ops)
        for group in self.ops.groups:
            subjects_to_hours = cp.deepcopy(self.ops.group_to_subjects[group])
            for week in range(self.ops.num_of_weeks):
                # days in week
                for day in range(5):
                    for class_num in range(self.ops.max_classes_per_day):
                        new_speciment.generate_class(subjects_to_hours, group, week, day, class_num)
        return new_speciment

    def add_random_speciment(self):
        self.speciments.append(self.get_random_speciment())

    def get_size(self):
        return len(self.speciments)



def save_speciment(speciment: Speciment):
    serialized = pickle.dumps(speciment.schedule_table, protocol=0)
    with open("schedule2", "wb") as f:
        f.write(serialized)

class Schedule:
    def __init__(self, ops: GeneralOptions):
        self.ops = ops
        self.best_speciment = Speciment(ops)


    def create(self, num_generations=1000, population_size=100, population_mutation_percentage=20,
               speciment_mutation_percentage=10, luckers_percentage=20, shake_percentage=90):
        total_num_of_generations = num_generations
        generation = self._create_random_generation(population_size)
        unchanged_streak = 0
        gen_number = 0
        start = datetime.now()
        x = []
        y = []
        while generation.get_best_speciment().evaluate() > self.ops.threshold and gen_number < total_num_of_generations:
            new_generation = generation.breed(generation.get_size()*4)
            new_generation.mutate(population_mutation_percentage, speciment_mutation_percentage)
            new_generation = new_generation.select(generation.get_size(), luckers_percentage)
            if generation.get_best_speciment().evaluate() - new_generation.get_best_speciment().evaluate() <\
                    self.ops.threshold:
                unchanged_streak += 1
                print(f"Unchanged streak:  {unchanged_streak}")
            else:
                unchanged_streak = 0
            if unchanged_streak == self.ops.change_threshold:
                new_generation.shake(shake_percentage)
                unchanged_streak = 0
            generation = new_generation
            if generation.get_best_speciment().evaluate() < self.best_speciment.evaluate():
                self.best_speciment = generation.get_best_speciment()
                save_speciment(self.best_speciment)
            print(f"Gen number: {gen_number} Evaluation: {generation.get_best_speciment().evaluate()}")
            x.append(gen_number)
            y.append(generation.get_best_speciment().evaluate())
            finish = datetime.now()
            gen_number += 1
            difference = finish - start
            time_spent = difference.seconds
            time_per_generation = time_spent // gen_number
            time_left = (total_num_of_generations - gen_number) * time_per_generation
            if gen_number % 10 == 0:
                print(f"Time spent: {time_spent}s\nTime left: {time_left}s")
                plt.ion()
                plt.show()
                plt.clf()
                plt.plot(x, y, label="Generation to evaluation")
                plt.legend()
                plt.draw()
                plt.pause(0.001)


    def get(self) -> Speciment:
        return self.best_speciment

    def _create_random_generation(self, num_of_speciments) -> Generation:
        generation = Generation(self.ops)
        for i in range(num_of_speciments):
            generation.add_random_speciment()
        return generation




def read_prof_file():
    groups_to_professors = dict()
    with open("professors.txt") as prof_file:
        lines = prof_file.read().splitlines()
        num_groups = int(lines[0])
        line_num = 1
        for i in range(num_groups):
            group = lines[line_num]
            line_num += 1
            num_profs = int(lines[line_num])
            line_num += 1
            profs = []
            for j in range(num_profs):
                prof = lines[line_num]
                line_num += 1
                profs.append(prof)
            groups_to_professors[group] = profs
    return groups_to_professors


def read_subj_file():
    groups_to_subjs = dict()
    with open("subjects.txt") as subjs_file:
        lines = subjs_file.read().splitlines()
        num_groups = int(lines[0])
        line_num = 1
        for i in range(num_groups):
            group = lines[line_num]
            line_num += 1
            num_subjs = int(lines[line_num])
            line_num += 1
            subjs_to_hours = dict()
            for j in range(num_subjs):
                subj_and_hours = lines[line_num].split(' ')
                line_num += 1
                subjs_to_hours[subj_and_hours[0]] = int(subj_and_hours[1])
            groups_to_subjs[group] = subjs_to_hours
    return groups_to_subjs


def read_teachers_file():
    teachers_to_subjs = dict()
    with open("teachers.txt") as teachers_file:
        lines = teachers_file.read().splitlines()
        num_teachers = int(lines[0])
        line_num = 1
        for i in range(num_teachers):
            teacher_and_num_subjs = lines[line_num].split(' ')
            line_num += 1
            teacher = teacher_and_num_subjs[0]
            num_subjs = int(teacher_and_num_subjs[1])
            subjs = []
            for j in range(num_subjs):
                subj = lines[line_num]
                line_num += 1
                subjs.append(subj)
            teachers_to_subjs[teacher] = subjs
    return teachers_to_subjs


if __name__ == '__main__':
    # with open("schedule", "rb") as fi:
    #     table = pickle.load(fi)

    ops = GeneralOptions(read_subj_file(), read_teachers_file(), read_prof_file(), weeks=14, max_classes_per_day=4,
                        groups=["k-15", "k-16", "k-17", "k-18"], small_rooms=[x for x in range(4)],
                        lect_rooms=[x for x in range(5, 7)])
    schedule = Schedule(ops)
    schedule.create()
    print(schedule.get().evaluate())
    serialized = pickle.dumps(schedule.get().schedule_table, protocol=0)
    with open("schedule", "wb") as f:
        f.write(serialized)
