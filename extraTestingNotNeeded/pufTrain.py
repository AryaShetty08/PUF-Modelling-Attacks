import numpy as np
import pypuf.io
import pypuf.simulation
import pypuf.attack
import pypuf.metrics

pufChallenges = pypuf.io.random_inputs(n=64, N=1, seed=42)
#print(np.unique(pufChallenges))

# use x = (1 - x) // 2. For the reverse conversion, use x = 1 - 2*x
# convertedPuf = (1 - pufChallenges) // 2

# ArbiterPUF simulation
arbPUF = pypuf.simulation.ArbiterPUF(n=64, seed=1)

# Challenge response set 
arbCRP = pypuf.io.ChallengeResponseSet.from_simulation(arbPUF, N=50000, seed=2)

pufResponses = [arbPUF.eval(pufChallenges) for _ in range(5)]

print("Challenge:", pufChallenges)
print("Responses:", [r[0] for r in pufResponses])

attack = pypuf.attack.LogisticRegressionAttack(arbCRP, seed=3, k=1, bs=1000, lr=0.001, epochs=100)

#arbCRP.save('arbCRP1.npz')
#crp_loaded = pypuf.io.ChallengeResponseSet.load('arbCRP1.npz')
#arbCRP == crp_loaded

#------------------
#attack
#attack = pypuf.attack.LRAttack2021(arbCRP, seed=3, k=4, bs=1000, lr=.001, epochs=100)
#attack.fit()

#model = attack.model
puf = pypuf.simulation.ArbiterPUF(n=64, noisiness=.25, seed=3)
#print(pypuf.metrics.reliability(puf, seed=3).mean())


#pypuf.metrics.similarity(arbPUF, model, seed=4)