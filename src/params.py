from dataclasses import dataclass

@dataclass
class ScanParams:
    b1_scaling: float
    larmor_frequency: float
    gx: float
    gy: float
    gz: float

    def __init(self, cons):
        self.read_params(cons)

    def read_params(self, cons):
        self.b1_scaling = cons.b1_scaling
        self.larmor_frequency = cons.larmor_frequency
        self.gx = cons.gradient_offset.x
        self.gy = cons.gradient_offset.y
        self.gz = cons.gradient_offset.z

    def write_params(self, cons):
        cons.b1_scaling = self.b1_scaling
        cons.larmor_frequency = self.larmor_frequency
        cons.gradient_offset.x = self.gx
        cons.gradient_offset.y = self.gy
        cons.gradient_offset.z = self.gz

    def print(self):
        print(f"b1_scaling: {self.b1_scaling}")
        print(f"larmor_frequency: {self.larmor_frequency}")
        print(f"gx: {self.gx}")
        print(f"gy: {self.gy}")
        print(f"gz: {self.gz}")
     

