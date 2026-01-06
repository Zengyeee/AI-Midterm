import genome
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np


class MotorType(Enum):
    PULSE = 1
    SINE = 2


class Motor:
    def __init__(self, control_waveform, control_amp, control_freq):
        self.motor_type = MotorType.PULSE if control_waveform <= 0.5 else MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        self.phase = 0

    def get_output(self):
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            return 1 if self.phase < np.pi else -1
        return float(np.sin(self.phase))


class Creature:
    def __init__(self, gene_count):
        self.spec = genome.Genome.get_gene_spec()
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count)

        self.flat_links = None
        self.exp_links = None
        self.motors = None

        self.start_position = None
        self.last_position = None

        # Fitness tracking
        self.spawn_z = None
        self.max_height_gain = 0.0

    def reset_evaluation(self):
        self.start_position = None
        self.last_position = None
        self.spawn_z = None
        self.max_height_gain = 0.0

    def update_position(self, pos):
        if self.start_position is None:
            self.start_position = pos
            self.spawn_z = float(pos[2])
        else:
            self.last_position = pos

    def update_max_height(self, z_value: float):
        if self.spawn_z is None:
            return
        gain = float(z_value) - float(self.spawn_z)
        if gain > self.max_height_gain:
            self.max_height_gain = gain

    def get_fitness(self) -> float:
        return float(self.max_height_gain)

    def get_flat_links(self):
        if self.flat_links is None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links

    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links

        exp_links = [self.flat_links[0]]
        genome.Genome.expandLinks(self.flat_links[0], self.flat_links[0].name, self.flat_links, exp_links)
        self.exp_links = exp_links
        return self.exp_links

    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")

        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))

        first = True
        for link in self.exp_links:
            if first:
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))

        robot_tag.setAttribute("name", "pepe")
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        self.get_expanded_links()
        if self.motors is None:
            motors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                motors.append(Motor(l.control_waveform, l.control_amp, l.control_freq))
            self.motors = motors
        return self.motors

    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0.0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        return float(np.linalg.norm(p1 - p2))

    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.reset_evaluation()
