

import ppo
ppo.ppo_algo(dict())
print('called ppo!!!')
print(ppo.__dict__)

class loller:
    mval = 8
    def __init__(self, val):
        self.val = val
master = {'hey':5, 'yo':2}
lolz1 = loller(master)
lolz2 = loller(master)

print(lolz1.val, lolz2.val)
master['hey'] = 20
print(lolz1.val, lolz2.val)
lolz1.val['hey'] = 50
print(lolz1.val, lolz2.val)
print('================')
lolz1.mval = 9
print(lolz1.mval, lolz2.mval)