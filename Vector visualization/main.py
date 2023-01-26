# visualize a very long vector

import torch
import matplotlib.pyplot as plt

# vec = torch.randint(1, 10, (10,))
# vec = torch.randn(10)
vec1 = torch.tensor(range(10))
vec = torch.randn(100).sort()[0]


def visualize(vecs):
    if not vecs:
        vec_copy = vecs.detach().clone()
        vec_copy = torch.cat((torch.zeros(1), vec_copy[:-1]))
        plt.plot(vecs, vec_copy)
    else:
        for vec in vecs:
            vec_copy = vec.detach().clone()
            vec_copy = torch.cat((torch.zeros(1), vec_copy[:-1]))
            plt.plot(vec, vec_copy)

    plt.show()


# refactorization = [[vec[i].item(), vec[i + 1].item()] for i in range(vec.shape[0] - 1)]


visualize((vec, vec1))

print("done")
