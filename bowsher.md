$$ R = \sum_i \sum_j w_{ij} f(x_i,x_j) $$

$$ \frac{\partial R}{\partial x_k} = \sum_j w_{kj} \frac{\partial f}{\partial x_1} (x_k,x_j) + \sum_i w_{ik} \frac{\partial f}{\partial x_2} (x_i,x_k) $$

$$ \frac{\partial R}{\partial x_k} = \sum_j w_{kj} \frac{\partial f}{\partial x_1} (x_k,x_j) + \sum_j w_{jk} \frac{\partial f}{\partial x_2} (x_j,x_k) $$

$$ \frac{\partial R}{\partial x_k} = \sum_j w_{kj} \frac{\partial f}{\partial x_1} (x_k,x_j) + w_{jk} \frac{\partial f}{\partial x_2} (x_j,x_k) $$