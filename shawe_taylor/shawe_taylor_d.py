
#Data class for shave-taylor algorithm. Mostly for return values.
from tools.states import FAILURE, SUCCESS

class ShaweTaylorD:

    def __init__(self, status=SUCCESS, prime=0, prime_seed=0, prime_gen_counter=0):
        self.status = status
        if self.status == FAILURE:
            self.prime = 0
            self.prime_seed = 0
            self.prime_gen_counter = 0
        else:
            self.prime = prime
            self.prime_seed = prime_seed
            self.prime_gen_counter = prime_gen_counter
        
    # def set_status(self, status):
    def print(self):
        if self.status == False:
            print("status: FAILURE")
        else:
            print("status: SUCCESS")
        print("prime: ", self.prime)
        print("prime_seed: ", self.prime_seed)
        print("prime_gen_counter: ", self.prime_gen_counter)


# st_return = STData(STData.FAILURE, 1, 2, 3)
# st_return.print()
