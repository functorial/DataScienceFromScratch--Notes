#
# Dimensionality Reduction
#

# Principal Component Analysis:
#   A technique used to extract one or more dimensions that
#   capture as much of the variation in the data as possible.
#       (1) Find the direction(s) which capture the greatest variance in data
#           Use gradient descent on gradient of variance!
#       (2) 

import numpy as np
from typing import List
from GradientDescent import gradient_step
import random
import tqdm

# Translate the data so that each dimension has mean 0
def de_mean(data: List[np.ndarray]) -> List[np.ndarray]:
    mean = np.mean(data, axis=0)
    return [v - mean for v in data]

# Get the direction of a vector
def direction_of(v: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(v)
    return v / mag

# Compute directional variance of `data` in the direction of `w`
def directional_variance(data: List[np.ndarray], w: np.ndarray) -> float:
    w_dir = direction_of(w)
    return np.sum(np.dot(v, w_dir) ** 2 / (len(w) - 1) for v in data)

# The gradient of f(w)=directional_variance(data,w)
# computed by hand
def directional_variance_gradient(data: List[np.ndarray], w: np.ndarray) -> np.ndarray:
    w_dir = direction_of(w)
    return np.array([sum([2*v[i]*np.dot(v,w_dir) for v in data]) / (len(w) - 1) for i in range(len(w))])

# First Principal Component:
#   The `w` which maximizes f(w)=directional_variance(data,w)

def first_principal_component(data: List[np.ndarray], steps: int=100, step_size: float=0.1) -> np.ndarray:
    # Starting point is random
    guess = np.array([random.uniform( np.min(data[0][i]), np.max(data[0][i]) ) for i in range(len(data[0]))])
    # Step from `guess` in the direction of `gradient`, `steps` times
    with tqdm.trange(steps) as t:
        for _ in t:
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient, step_size)
            # This hopefully shows that `dv` is increasing over steps
            t.set_description(f"dv: {dv:.3f}")

    return direction_of(guess)

# Now we need to project the data onto the FPC axis
def project(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Projects v onto w"""
    return np.dot(v, w) * (w / np.linalg.norm(w))

# If we want to find more principal components,
# Subtract the first component and run again
def remove_prjection_from_vector(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return v - project(v, w)

def remove_projection(data: List[np.ndarray], w: np.ndarray) -> List[np.ndarray]:
    return [remove_prjection_from_vector(v, w) for v in data]

# Define a function which iterates this process
def pca(data: List[np.ndarray], num_components: int) -> List[np.ndarray]:
    components: List[np.ndarray]=[]
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)
    return components

# Now we want to transform our data into the lower-dimensional space
# spanned by the components
def transform_vector(v: np.ndarray, components: List[np.ndarray]) -> np.ndarray:
    return np.array([np.dot(v, w) for w in components])

def transform(data: List[np.ndarray], components: List[np.ndarray]) -> List[np.ndarray]:
    return [transform_vector(v, components) for v in data]


# Note:
#   Principal components are also given by the eigenvectors of the covariance matrix.
#       (This can be proven by an 'easy' computation, pushing definitions)
#   Thus the principal component decomposition is the same as a spectral decomp on the covariance matrix

def sample_covariance_matrix(data: List[np.ndarray]) -> np.ndarray:
    return np.cov(data)

from typing import Tuple
def spectral_decomposition(m: np.ndarray) -> Tuple:
    """Gives (eigenvalues, eigen direction vectors)"""
    return np.linalg.eig(m)

def pca_2(data: List[np.ndarray], num_components: int):
    evalues, components = spectral_decomposition(sample_covariance_matrix(data))
    # sort the components by largest to smallest evalue
    components_sorted_by_evalue = [e[1] for e in sorted(zip(evalues, components), reverse=True)]
    # the nth PC is the eigenvector corresp to the nth largest eigenvalue
    return components_sorted_by_evalue[:num_components]
    


# Test on some data!!!
#
from matplotlib import pyplot as plt 

# gather data
data = []
for _ in range(50):
    x = random.random()
    y = x + random.random()    # Feel free to adjust this function
                                    # The more non-linear the data is, the more information is lost in the PCA
    data.append(np.array([x, y]))
# demean the data (lol)
data = de_mean(data)
# compute pricipal components
fpcs = pca(data, 2)
print(fpcs[0])
# project data onto first principal component
data_proj1 = remove_projection(data, fpcs[1])
data_proj2 = remove_projection(data, fpcs[0])

# set up a plot
xs0 = [data[i][0] for i in range(len(data))]
ys0 = [data[i][1] for i in range(len(data))]
xs1 = [data_proj1[i][0] for i in range(len(data))]
ys1 = [data_proj1[i][1] for i in range(len(data))]
xs2 = [data_proj2[i][0] for i in range(len(data))]
ys2 = [data_proj2[i][1] for i in range(len(data))]

fig, ax = plt.subplots(1, 2)

ax[0].scatter(xs0, ys0, marker='.')
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)
ax[0].quiver(np.mean(xs0), np.mean(ys0), fpcs[0][0], fpcs[0][1], angles='xy', scale_units='xy', scale=3)
ax[0].quiver(np.mean(xs0), np.mean(ys0), fpcs[1][0], fpcs[1][1], angles='xy', scale_units='xy', scale=3)

ax[1].scatter(xs1, ys1, marker='.', color='red')
ax[1].scatter(xs2, ys2, marker='.')

ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[1].quiver(np.mean(xs0), np.mean(ys0), fpcs[0][0], fpcs[0][1], angles='xy', scale_units='xy', scale=3)
ax[1].quiver(np.mean(xs0), np.mean(ys0), fpcs[1][0], fpcs[1][1], angles='xy', scale_units='xy', scale=3)

plt.title('Dimensionality Reduction')
plt.show()